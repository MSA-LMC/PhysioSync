import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
def constrast_loss(x, criterion, tau):
    LARGE_NUM = 1e9
    x = F.normalize(x, dim=-1)
    num = int(x.shape[0]/2)
    hidden1, hidden2 = torch.split(x, num)
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = torch.arange(0,num).to('cuda')
    masks = F.one_hot(torch.arange(0,num), num).to('cuda')
    logits_aa = torch.matmul(hidden1, hidden1_large.T) / tau
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = torch.matmul(hidden2, hidden2_large.T) / tau
    logits_bb = logits_bb - masks * LARGE_NUM
    logits_ab = torch.matmul(hidden1, hidden2_large.T) / tau
    logits_ba = torch.matmul(hidden2, hidden1_large.T) / tau
    loss_a = criterion(torch.cat([logits_ab, logits_aa], 1),
                       labels)
    loss_b = criterion(torch.cat([logits_ba, logits_bb], 1),
                       labels)
    loss = torch.mean(loss_a + loss_b)
    return loss, labels, logits_ab

class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None, relative_position_bias=None):
        B, N, C = x.shape

        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q.float() @ k.float().transpose(-2, -1))

        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1).type_as(x)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            with_mixffn=False,
            layer_scale_init_values=0.1,
            max_text_len=40,
            prompt=False,
            prompt_len=2,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.mlp_vl = None
        if with_mixffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)

        self.gamma_1 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0
        self.gamma_2 = \
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True) \
                if layer_scale_init_values is not None else 1.0

        self.max_text_len = max_text_len
        #
        self.prompt_len = prompt_len
        self.eeg_prompt = nn.Parameter(torch.zeros((1, prompt_len, dim)),
                                       requires_grad=True) if prompt is True else None
        self.ecg_prompt = nn.Parameter(torch.zeros((1, prompt_len, dim)),
                                       requires_grad=True) if prompt is True else None
        if self.eeg_prompt is not None:
            trunc_normal_(self.eeg_prompt, std=0.02)
        if self.ecg_prompt is not None:
            trunc_normal_(self.ecg_prompt, std=0.02)

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        if modality_type == "eeg" and self.eeg_prompt is not None:
            eeg_prompt = self.eeg_prompt.expand(x.size()[0], -1, -1) if self.eeg_prompt is not None else None
            x = torch.cat((x, eeg_prompt), dim=1)
        elif modality_type == "ecg" and self.ecg_prompt is not None:
            ecg_prompt = self.ecg_prompt.expand(x.size()[0], -1, -1) if self.ecg_prompt is not None else None
            x = torch.cat((x, ecg_prompt), dim=1)
        x = x + self.drop_path(
            self.gamma_1 * self.attn(self.norm1(x), mask=mask, relative_position_bias=relative_position_bias))

        if modality_type == "ecg":
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        elif modality_type == "eeg":
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        if modality_type == "eeg" and self.eeg_prompt is not None:
            x = x[:, :-self.prompt_len, :]
        elif modality_type == "ecg" and self.ecg_prompt is not None:
            x = x[:, :-self.prompt_len, :]

        return x


class MultiViewEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim, heads):
        super().__init__()
        self.output_dim = output_dim
        self.heads = heads

        self.transform1 = nn.Linear(input_dim, output_dim)
        self.transform2 = nn.Linear(input_dim, output_dim * heads)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = torch.flatten(x,1)
        B, _ = x.size()
        x1 = self.transform1(x).unsqueeze(1).repeat(1, self.heads, 1)
        x2 = self.sigmoid(self.transform2(x))
        x2 = x2.reshape(B, self.heads, self.output_dim)

        x = torch.mul(x1, x2)
        x = self.bn(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Encoder_eeg(nn.Module):

    def __init__(self, embed_dim=128, eeg_dim=1792, mlp_dim=512, eeg_seq_len=3, use_abs_pos_emb=False, num_heads=4,
                 depth=3,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 layer_scale_init_values=0.1,
                 mixffn_start_layer_index=2,
                 use_mean_pooling=False,
                 prompt=True,
                 prompt_len=2,):
        super(Encoder_eeg, self).__init__()
        self.eeg_seq_len = eeg_seq_len
        self.use_abs_pos_emb = use_abs_pos_emb
        self.pos_drop = nn.Dropout(p=0.2)
        self.eeg_transform = MultiViewEmbedding(eeg_dim, embed_dim, eeg_seq_len)
        self.eeg_pos_embed = nn.Parameter(torch.zeros(1, eeg_seq_len + 1, embed_dim)) if not self.use_abs_pos_emb else None
        self.eeg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = norm_layer(embed_dim)
        self.mixffn_start_layer_index = mixffn_start_layer_index
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        if self.eeg_pos_embed is not None:
            trunc_normal_(self.eeg_pos_embed, std=0.02)
        trunc_normal_(self.eeg_cls_token, std=0.02)
        trunc_normal_(self.eeg_type_embed, std=0.02)
        self.apply(self._init_weights)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_mixffn=(i >= self.mixffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=eeg_seq_len + 1,
                    prompt=prompt,
                    prompt_len=prompt_len
                )
                for i in range(depth)
            ]
        )
        self.fc_projector1 = nn.Linear(mlp_dim, 1024)
        self.bn_fc_p1 = nn.BatchNorm1d(1024)
        self.fc_projector2 = nn.Linear(1024, 2048)
        self.bn_fc_p2 = nn.BatchNorm1d(2048)
        self.fc_projector3 = nn.Linear(2048, 4096)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mode):
        modality_type = 'eeg'
        x = self.eeg_transform(x)
        eeg_cls_tokens = self.eeg_cls_token.expand(x.size()[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((eeg_cls_tokens, x), dim=1)
        x = x + self.eeg_type_embed.expand(x.size()[0], x.size()[1], -1)
        if self.eeg_pos_embed is not None:
            x = x + self.eeg_pos_embed.expand(x.size()[0], -1, -1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, modality_type=modality_type)
        x = self.norm(x)
        x_encoder = torch.flatten(x, 1)
        if mode == 'contrast':
            x = self.fc_projector1(x_encoder)
            x = self.bn_fc_p1(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector2(x)
            x = self.bn_fc_p2(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector3(x)
            return x, x_encoder
        return x_encoder


class Encoder_ecg(nn.Module):

    def __init__(self, embed_dim=128, ecg_dim=256, mlp_dim=512, ecg_seq_len=3, use_abs_pos_emb=False, num_heads=4,
                 depth=3,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm,
                 layer_scale_init_values=0.1,
                 mixffn_start_layer_index=2,
                 use_mean_pooling=False,
                 prompt=False,
                 prompt_len=2):
        super(Encoder_ecg, self).__init__()
        self.pos_drop = nn.Dropout(p=0.2)
        self.ecg_seq_len = ecg_seq_len
        self.use_abs_pos_emb = use_abs_pos_emb
        self.ecg_transform = MultiViewEmbedding(ecg_dim, embed_dim, ecg_seq_len)
        self.ecg_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ecg_type_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.ecg_pos_embed = nn.Parameter(
            torch.zeros(1, ecg_seq_len + 1, embed_dim)) if not self.use_abs_pos_emb else None
        self.mixffn_start_layer_index = mixffn_start_layer_index
        self.norm = norm_layer(embed_dim)
        if self.ecg_pos_embed is not None:
            trunc_normal_(self.ecg_pos_embed, std=0.02)
        trunc_normal_(self.ecg_cls_token, std=0.02)
        trunc_normal_(self.ecg_type_embed, std=0.02)
        self.apply(self._init_weights)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_mixffn=(i >= self.mixffn_start_layer_index),
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=ecg_seq_len + 1,
                    prompt=prompt,
                    prompt_len=prompt_len
                )
                for i in range(depth)
            ]
        )
        self.fc_projector1 = nn.Linear(mlp_dim, 1024)
        self.bn_fc_p1 = nn.BatchNorm1d(1024)
        self.fc_projector2 = nn.Linear(1024, 2048)
        self.bn_fc_p2 = nn.BatchNorm1d(2048)
        self.fc_projector3 = nn.Linear(2048, 4096)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, mode):
        modality_type = 'ecg'
        x = self.ecg_transform(x)
        ecg_cls_tokens = self.ecg_cls_token.expand(x.size()[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((ecg_cls_tokens, x), dim=1)
        x = x + self.ecg_type_embed.expand(x.size()[0], x.size()[1], -1)
        if self.ecg_pos_embed is not None:
            x = x + self.ecg_pos_embed.expand(x.size()[0], -1, -1)
        for blk in self.blocks:
            x = blk(x, modality_type=modality_type)
        x = self.norm(x)
        x_encoder = torch.flatten(x, 1)
        if mode == 'contrast':
            x = self.fc_projector1(x_encoder)
            x = self.bn_fc_p1(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector2(x)
            x = self.bn_fc_p2(x)
            x = F.dropout(x, p=0.2)
            x = self.fc_projector3(x)
            return x, x_encoder
        return x_encoder


class LinearLayer(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, x):
        x = self.clf(x)
        return x


class CMCon(nn.Module):

    def __init__(self, mlp_dim, out_dim=1024):
        super(CMCon, self).__init__()
        self.layers = LinearLayer(mlp_dim, out_dim)
        self.constrast_loss = constrast_loss
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, eeg, ecg, criterion, tau):

        eeg = self.layers(eeg)
        eeg = self.bn(eeg)
        ecg = self.layers(ecg)
        ecg = self.bn(ecg)

        half_size = eeg.shape[0] // 2
        temp = eeg[:half_size, :].clone()
        eeg[:half_size, :] = ecg[:half_size, :]
        ecg[:half_size, :] = temp

        tem_loss_1, _, _ = constrast_loss(eeg, criterion, tau)
        tem_loss_2, _, _ = constrast_loss(ecg, criterion, tau)

        loss = 0.5 * tem_loss_1 + 0.5 * tem_loss_2
        return loss


class ClassificationSubNetwork(nn.Module):
    def __init__(self,
                 mlp_dim,
                 num_classes):
        super(ClassificationSubNetwork, self).__init__()
        self.layers = nn.ModuleList([LinearLayer(mlp_dim, num_classes)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class classifier_max(nn.Module):
    def __init__(self, num_classes, mlp_dim=512):
        super(classifier_max, self).__init__()

        self.Classification = ClassificationSubNetwork(mlp_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

        self.relu = nn.ReLU(inplace=True)

        self.fc_classifier1 = nn.Linear(1024, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)

        self.fc_classifier2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)

        self.fc_classifier3 = nn.Linear(128, num_classes)

    def forward(self, eeg, gsr):

        TCPLogit_eeg = self.Classification(eeg)
        TCPLogit_gsr = self.Classification(gsr)
        pred_eeg = F.softmax(TCPLogit_eeg, dim=1)
        pred_gsr = F.softmax(TCPLogit_gsr, dim=1)
        pred_eeg = torch.max(pred_eeg, dim=1)[0][:, None]
        pred_gsr = torch.max(pred_gsr, dim=1)[0][:, None]

        eeg = eeg * pred_eeg
        gsr = gsr * pred_gsr

        x = torch.cat([eeg, gsr], dim=1)
        x = self.fc_classifier1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.fc_classifier2(x)
        x = self.bn_fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_classifier3(x)
        return x









