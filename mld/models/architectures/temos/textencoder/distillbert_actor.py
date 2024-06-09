from .distillbert import DistilbertEncoderBase
import torch
import os
from typing import List, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt
from mld.models.operator import PositionalEncoding
from mld.utils.temos_utils import lengths_to_mask
from torch.nn.functional import normalize


class DistilbertActorAgnosticEncoder(DistilbertEncoderBase):
    def __init__(self, modelpath: str,
                 finetune: bool = False,
                 vae: bool = True,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__(modelpath=modelpath, finetune=finetune)
        self.save_hyperparameters(logger=False)

        encoded_dim = self.text_encoded_dim

        # Projection of the text-outputs into the latent space
        self.projection = nn.Sequential(nn.ReLU(),
                                        nn.Linear(encoded_dim, latent_dim))

        # TransformerVAE adapted from ACTOR
        # Action agnostic: only one set of params
        if vae:
            self.mu_token = nn.Parameter(torch.randn(latent_dim))
            self.logvar_token = nn.Parameter(torch.randn(latent_dim))
        else:
            self.emb_token = nn.Parameter(torch.randn(latent_dim))

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_encoder_layer = nn.TransformerEncoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransEncoder = nn.TransformerEncoder(seq_trans_encoder_layer,
                                                     num_layers=num_layers)

    def forward(self, texts: List[str]) -> Union[Tensor, Distribution]:
        text_encoded, mask = self.get_last_hidden_state(texts, return_mask=True)

        x = self.projection(text_encoded)
        bs, nframes, _ = x.shape
        # bs, nframes, totjoints, nfeats = x.shape
        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]

        if self.hparams.vae:
            mu_token = torch.tile(self.mu_token, (bs,)).reshape(bs, -1)
            logvar_token = torch.tile(self.logvar_token, (bs,)).reshape(bs, -1)

            # adding the distribution tokens for all sequences
            xseq = torch.cat((mu_token[None], logvar_token[None], x), 0)

            # create a bigger mask, to allow attend to mu and logvar
            token_mask = torch.ones((bs, 2), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)
        else:
            emb_token = torch.tile(self.emb_token, (bs,)).reshape(bs, -1)

            # adding the embedding token for all sequences
            xseq = torch.cat((emb_token[None], x), 0)

            # create a bigger mask, to allow attend to emb
            token_mask = torch.ones((bs, 1), dtype=bool, device=x.device)
            aug_mask = torch.cat((token_mask, mask), 1)

        # add positional encoding
        xseq = self.sequence_pos_encoding(xseq)
        final = self.seqTransEncoder(xseq, src_key_padding_mask=~aug_mask)

        if self.hparams.vae:
            mu, logvar = final[0], final[1]
            std = logvar.exp().pow(0.5)
            # https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
            try:
                dist = torch.distributions.Normal(mu, std)
            except ValueError:
                import ipdb; ipdb.set_trace()  # noqa
                pass
            return dist
        else:
            return final[0]
    
    # compute score for retrieval
    def compute_scores(self, texts, unit_embs=None, embs=None):
        # not both empty
        assert not (unit_embs is None and embs is None)
        # not both filled
        assert not (unit_embs is not None and embs is not None)

        output_str = False
        # if one input, squeeze the output
        if isinstance(texts, str):
            texts = [texts]
            output_str = True

        # compute unit_embs from embs if not given
        if unit_embs is not None:
            unit_embs = normalize(unit_embs,p=2, dim=1)
        # texts=[]
        # import pdb; pdb.set_trace()
          
        # idlist = [i for i in range(len(unit_embs))]
        # keyids = np.array(idlist)   
        # for keyid in keyids:   
        #     idstr = "%05d"%(keyid)
            
        #     with open(os.path.join("/comp_robot/lushunlin/motion-latent-diffusion/retrieval/test_text", idstr+".txt"), "r") as f:
        #         text = f.read()
        #         texts.append(text)
                
        with torch.no_grad():
            latent_unit_texts = normalize(self(texts).loc).to(unit_embs.device)
            # np.save('text_embedding.npy', latent_unit_texts.cpu().numpy())
            # txt_emb = torch.tensor(np.load("/comp_robot/lushunlin/motion-latent-diffusion/experiments/temos/temos_humanml3d_kl_1e-5_wlatent_infonce_4gpu_nce_1e-1/embeddings/val/epoch_99/text_embedding.npy"))[2007].unsqueeze(0).cuda()
            # compute cosine similarity between 0 and 1
            # inner_product = unit_embs @ txt_emb.T
            # import pdb; pdb.set_trace()
            inner_product = unit_embs @ latent_unit_texts.T
            scores = inner_product.T/2 + 0.5
            scores = scores.cpu().numpy()
            # plt.matshow(scores, cmap=plt.cm.Reds)#这里设置颜色为红色，也可以设置其他颜色
            # plt.title("matrix A")
            # plt.show()
            # plt.savefig("t.png")
        if output_str:
            scores = scores[0]

        return scores