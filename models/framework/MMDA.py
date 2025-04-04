import math

import torch
import torch.nn.functional as F
from torch import nn, einsum

from models.layer.MoE.stMoE.st_moe import GEGLU, TopNGating, MixtureOfExpertsReturn, Experts
from models.network.clip import clip

"""
MoE adaptor + diff attn + distributed label + subspace(U cross)
"""

spoof_templates = [
    'This is a photo of a spoof face',
    'This is a photo of an attack face',
    'This is a photo of a fake face',
    'This is a photo of a non-real face',
    'This is a photo of a counterfeit face',
    'This is a photo of a printed spoof face',
    'This is a photo of an adversarial patch face',
    'This is a photo of a wearable adversarial sticker face',
    'This is a photo of a printed adversarial image face',
    'This is a photo of a physical impersonation attack face',
    'This is a photo of an adversarial perturbation face',
    'This is a photo of an adversarial makeup face',
    'This is a photo of a facial attribute modification face',
    'This is a photo of an adversarial face image',
    'This is a photo of a physical-world printed attack face',
    'This is a photo of a synthetic image face',
    'This is a photo of an adversarial face mask',
    'This is a photo of a semantic adversarial attack face',
    'This is an example of a spoof face',
    'This is an example of an attack face',
    'This is not a real face',
    'This is how a spoof face looks like',
    'a photo of a spoof face',
    'a printout shown to be a spoof face',
]

real_templates = [
    'This is a photo of a real face',
    'This is a photo of an authentic face',
    'This is a photo of a vital face',
    'This is a photo of a biometric face',
    'This is a photo of a genuine face',
    'This is a photo of an alive face',
    'This is a photo of a real-time face',
    'This is a photo of a vivacious face',
    'This is a photo of a veritable face',
    'This is a photo of a valid face',
    'This is a photo of an actual face',
    'This is a photo of an existent face',
    'This is a photo of a true face',
    'This is a photo of a legitimate face',
    'This is a photo of a natural face',
    'This is a photo of an unfeigned face',
    'This is a photo of an unimpeachable face',
    'This is a photo of a credible face'
    'This is an example of a real face',
    'This is a bonafide face',
    'This is a real face',
    'This is how a real face looks like',
    'a photo of a real face',
    'This is not a spoof face',
]


class MultiHeadDomainDiffAttention(nn.Module):
    def __init__(self, d_model=512, n_head=8, dropout=0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head

        self.temperature = d_model ** 0.5

        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.w_dqs = nn.Linear(d_model, d_model, bias=False)
        self.w_dks = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * 12)
        self.lambda_q1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(d_model, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.ln = RMSNorm(d_model, eps=1e-5)

    def forward(self, q, k, v, domain_q, domain_k):
        n_head = self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_d = domain_q.size(1)

        q = self.w_qs(q).view(sz_b, len_q, n_head, -1)  # [b,N,h,d]
        k = self.w_ks(k).view(sz_b, len_k, n_head, -1)  # [b,N,h,d]
        v = self.w_vs(v).view(sz_b, len_v, n_head, -1)  # [b,N,h,d]
        dq = self.w_dqs(domain_q).view(sz_b, len_d, n_head, -1)  # [b,N,h,d]
        dk = self.w_dks(domain_k).view(sz_b, len_d, n_head, -1)  # [b,N,h,d]

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [b,h,N,d]
        dq, dk = dq.transpose(1, 2), dk.transpose(1, 2)

        a = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [b,h,N,N]
        W = self.dropout(F.softmax(a, dim=-1))

        da = torch.matmul(dq / self.temperature, dk.transpose(2, 3))
        dW = self.dropout(F.softmax(da, dim=-1))

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        W = W - lambda_full * dW

        out = torch.matmul(W, v)  # [b,h,N,d]
        out = out.transpose(1, 2).contiguous().view(sz_b, len_q, -1)  # [b,N,d]
        out = self.ln(out)
        out = out * (1 - self.lambda_init)

        return out

class ConstExpert(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.const = nn.Parameter(torch.empty(dim).normal_(std=0.02))

    def forward(self, x):
        return self.const.expand(x.size(0), x.size(1), -1)


class Expert(nn.Module):
    def __init__(self, dim=512, hidden_mult=4):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden),
            nn.Linear(dim_hidden, dim)
        )

    def forward(self, x):
        return self.net(x)


class MoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts=4,
                 expert_hidden_mult=4,
                 threshold_train=0.2,
                 threshold_eval=0.2,
                 capacity_factor_train=1.25,
                 capacity_factor_eval=2.,
                 gating_top_n=2,
                 balance_loss_coef=1e-2,
                 router_z_loss_coef=1e-3,
                 straight_through_dispatch_tensor=True,
                 differentiable_topk=False,
                 differentiable_topk_fused=True,
                 is_distributed=None,
                 allow_var_seq_len=False
                 ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts

        # experts = [copy.deepcopy(mlp) for _ in range(num_experts)]
        experts = [
            Expert(dim=dim),
            Expert(dim=dim),
            Expert(dim=dim),
            Expert(dim=dim),
            ConstExpert(dim=dim),
            ConstExpert(dim=dim),
            ConstExpert(dim=dim),
        ]

        self.gate = TopNGating(
            dim,
            top_n=gating_top_n,
            num_gates=len(experts),
            straight_through_dispatch_tensor=straight_through_dispatch_tensor,
            differentiable_topk=differentiable_topk,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval
        )

        self.experts = Experts(
            experts,
            is_distributed=is_distributed,
            allow_var_seq_len=allow_var_seq_len
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
            self,
            x,
            noise_gates=False,
            noise_mult=1.
    ):
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss = self.gate(x, noise_gates=noise_gates,
                                                                                 noise_mult=noise_mult)

        # dispatch

        expert_inputs = einsum('b n d, b n e c -> b e c d', x, dispatch_tensor)

        # feed the expert inputs through the experts.

        expert_outputs = self.experts(expert_inputs)

        # combine

        output = einsum('b e c d, b n e c -> b n d', expert_outputs, combine_tensor)

        # losses

        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses

        total_aux_loss = weighted_balance_loss + weighted_router_z_loss

        return MixtureOfExpertsReturn(output, total_aux_loss, balance_loss, router_z_loss)


class Fusion(nn.Module):
    def __init__(self, dim=512, num_modal=3):
        super().__init__()
        self.attn = MultiHeadDomainDiffAttention(d_model=dim, n_head=16)
        self.net = MoE(dim=dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim * num_modal, dim)

    def forward(self, x, d_x):
        x1 = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), self.ln1(d_x), self.ln1(d_x))
        x2, aux_loss, _, _ = self.net(self.ln2(x1))
        x = x + x1 + x2

        b, _, _ = x.size()
        x = x.view(b, 1, -1)
        x = self.proj(x)
        return x, aux_loss


class FeatureSpaceRefine(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.num_head = 16
        assert dim % self.num_head == 0
        self.attn = nn.MultiheadAttention(embed_dim=512, num_heads=self.num_head)
        self.to_feature = MoE(dim=dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):  # [b,1,d]
        b, _, d = x.size()
        x1 = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x2, aux_loss, _, _ = self.to_feature(self.ln2(x1))
        x = x + x1 + x2
        return x, aux_loss


class Classifier(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.net = MoE(dim=dim)
        self.proj = nn.Linear(dim, 2)

    def forward(self, x):  # [b,1,d]
        x, aux_loss, _, _ = self.net(x)
        x = self.proj(x).squeeze(1)
        return x, aux_loss


class U_DSA(nn.Module):
    def __init__(self,num_layer=3,dim=512):
        super().__init__()
        self.refine_layer = num_layer
        self.refine = nn.ModuleList([FeatureSpaceRefine(dim=dim) for _ in range(self.refine_layer)])
        self.remapping = nn.ModuleList([FeatureSpaceRefine(dim=dim) for _ in range(self.refine_layer)])
    def forward(self,e,spoof,real):
        es = [e]
        spoofs = [spoof]
        reals = [real]
        if self.refine_layer == 0:
            return es,spoofs,reals,None
        aux_loss = []
        for l in range(self.refine_layer):
            e, aux_loss1 = self.refine[l](es[-1])
            spoof, aux_loss2 = self.refine[l](spoofs[-1])
            real, aux_loss3 = self.refine[l](reals[-1])
            es.append(e)
            spoofs.append(spoof)
            reals.append(real)
            aux_loss.append(aux_loss1)
            aux_loss.append(aux_loss2)
            aux_loss.append(aux_loss3)
        for l in reversed(range(1,self.refine_layer + 1)):
            e, aux_loss1 = self.remapping[l-1](es[l])
            es.append(e + es[l-1])
            spoofs.append(spoofs[l-1])
            reals.append(reals[l-1])
            aux_loss.append(aux_loss1)
        aux_loss = torch.sum(torch.stack(aux_loss,dim=0),dim=0)
        return es,spoofs,reals,aux_loss

class adaptor(nn.Module):
    def __init__(self, dim=512, num_modal=3):
        super().__init__()
        self.fusion = Fusion(dim=dim, num_modal=num_modal)
        self.classifier = Classifier(dim=dim)
        self.u_align = U_DSA(dim=dim, num_layer=7)

    def forward(self, xs, domain_xs, spoof_embedding, real_embedding):
        x = torch.stack(xs, dim=1)  # [b,m,d]
        d_x = torch.stack(domain_xs, dim=1)
        aux_loss = []
        e, aux_loss1 = self.fusion(x, d_x)  # [b,1,d]
        aux_loss.append(aux_loss1)

        es,spoofs,reals,aux_loss2 = self.u_align(e,spoof_embedding,real_embedding)
        if aux_loss2 is not None:
            aux_loss.append(aux_loss2)

        classify, aux_loss3 = self.classifier(es[-1])
        aux_loss.append(aux_loss3)
        if self.training:
            spoof_classify, aux_loss4 = self.classifier(spoof_embedding)
            spoof_class_label = torch.tensor([0.9, 0.1]).expand(spoof_classify.size(0), -1).cuda()
            loss_spoof = F.cross_entropy(spoof_classify, spoof_class_label)
            real_classify, aux_loss5 = self.classifier(real_embedding)
            real_class_label = torch.tensor([0.1, 0.9]).expand(real_classify.size(0), -1).cuda()
            loss_real = F.cross_entropy(real_classify, real_class_label)
            aux_loss.append(aux_loss4 + aux_loss5 + loss_real + loss_spoof)

        aux_loss = torch.sum(torch.stack(aux_loss,dim=0),dim=0)

        return es, classify, spoofs, reals, aux_loss


class MMDA(nn.Module):

    def __init__(self):
        super(MMDA, self).__init__()
        self.model, _ = clip.load("ViT-B/16", 'cuda:0')
        self.model.to(torch.float32)
        self.dtype = torch.float32
        """把positional embedding改成3模态"""
        self.out = None
        self.adapter = adaptor()

        hidden_mult = 4
        dim = 512
        dim_hidden = int(dim * hidden_mult * 2 / 3)
        self.classifier = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden),
            nn.Linear(dim_hidden, 2)
        )

        for p in self.model.parameters(): p.require_grads = False

    def forward_visual(self, x):
        B, C, H, W = x.shape

        x = self.model.visual.conv1(x.type(self.dtype))

        x = x.reshape(B, x.shape[1], -1)

        x = x.permute(0, 2, 1)

        x = torch.cat((self.model.visual.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype,
                                                                                   device=x.device), x), dim=1)
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.visual.ln_post(x[:, 0, :])

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj

        return x

    def cal_logits(self, visual_embeddings, textual_embeddings):  # [b,m,d] [b,n,d]
        eps = 1e-20
        visual_embeddings = visual_embeddings / (visual_embeddings.norm(dim=-1, keepdim=True) + eps)
        textual_embeddings = textual_embeddings / (textual_embeddings.norm(dim=-1, keepdim=True) + eps)

        # cosine similarity as logits
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * visual_embeddings @ textual_embeddings.transpose(-1, -2)  # [B, m,n]
        logits_per_image = logits_per_image.max(-1).values  # [B,m]

        return logits_per_image

    def forward(self, input_1, input_2, input_3, domain=None):
        spoof_texts = clip.tokenize(spoof_templates).cuda(non_blocking=True)  # tokenize
        real_texts = clip.tokenize(real_templates).cuda(non_blocking=True)  # tokenize

        spoof_class_embeddings = self.model.encode_text(spoof_texts).cuda()  # [x1,d]
        real_class_embeddings = self.model.encode_text(real_texts).cuda()  # [x2,d]

        rgb_image_features = self.forward_visual(input_1)  # [B, 512]
        depth_image_features = self.forward_visual(input_2)  # [B, 512]
        ir_image_features = self.forward_visual(input_3)  # [B, 512]

        always_random_select = True
        if self.training or always_random_select:
            domain_rgb_image_features = []
            domain_depth_image_features = []
            domain_ir_image_features = []
            for b1 in range(input_1.size(0)):
                for b2 in range(input_1.size(0)):
                    if domain[b1] == domain[b2]:
                        domain_rgb_image_features.append(rgb_image_features[b2])
                        domain_depth_image_features.append(depth_image_features[b2])
                        domain_ir_image_features.append(ir_image_features[b2])
                        break
            domain_rgb_image_features = torch.stack(domain_rgb_image_features, dim=0)
            domain_depth_image_features = torch.stack(domain_depth_image_features, dim=0)
            domain_ir_image_features = torch.stack(domain_ir_image_features, dim=0)
        else:
            domain_rgb_image_features = rgb_image_features
            domain_depth_image_features = depth_image_features
            domain_ir_image_features = ir_image_features

        es, classify, spoof_embeddings, real_embeddings, self.total_loss = self.adapter(
            [rgb_image_features, depth_image_features, ir_image_features],
            [domain_rgb_image_features, domain_depth_image_features,
             domain_ir_image_features], spoof_class_embeddings.unsqueeze(1),
            real_class_embeddings.unsqueeze(1))  # [b,1,d]

        # cal e logits
        e_logits_per_images = []
        for i in range(len(es)):
            e_real_logits_per_image = self.cal_logits(es[i],real_embeddings[i].squeeze(1).unsqueeze(0).expand(es[i].size(0), -1, -1))
            e_spoof_logits_per_image = self.cal_logits(es[i],spoof_embeddings[i].squeeze(1).unsqueeze(0).expand(es[i].size(0), -1, -1))
            e_logits_per_image = torch.cat([e_spoof_logits_per_image, e_real_logits_per_image], dim=-1)  # [b,2]
            e_logits_per_images.append(e_logits_per_image)


        # print(image_features, text_features, similarity)
        self.out = {
            # "mix": e1_logits_per_image.narrow(0, 0, input_1.size(0)),
            # "align": e2_logits_per_image.narrow(0, 0, input_1.size(0)),
            "cls": classify}
        self.aligns = e_logits_per_images
        for i in range(len(e_logits_per_images)):
            self.out[f"align{i}"] = e_logits_per_images[i].narrow(0, 0, input_1.size(0))
        return self.out

    def cal_loss(self, spoof_label, loss_func):
        loss_func = nn.CrossEntropyLoss()
        # logits [b,2]
        # label [b,1]现在是对称平滑
        one_hot_label = F.one_hot(spoof_label.squeeze(-1), num_classes=2)
        smoothing = 0.1
        smooth_labels = (1.0 - smoothing) * one_hot_label + smoothing
        # loss_mix = loss_func(self.out['mix'], smooth_labels)
        loss_align = []
        for i in range(len(self.aligns)):
            loss_align_i = loss_func(self.aligns[i], smooth_labels)
            loss_align.append(loss_align_i)
        loss_align = torch.sum(torch.stack(loss_align,dim=0),dim=0)
        loss_classify = loss_func(self.out['cls'], smooth_labels)

        total_loss = self.total_loss + loss_classify + loss_align

        loss = {
            'total_loss': total_loss,
        }
        # print(loss)
        return loss


if __name__ == '__main__':
    # clip_anymodal_moe()
    x = torch.randn(3, 3, 224, 224).cuda()
    label = torch.randint(0, 1, (3, 1)).cuda()

    model = MMDA().cuda()
    out = model(x, x, x, label)
    model.cal_loss(label,nn.functional.cross_entropy)
    # mask = torch.rand(12, 2, 4, 6) < 0.5
