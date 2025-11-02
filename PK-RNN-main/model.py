import torch
import torch.nn as nn
from torch.optim import SGD


class EHREmbeddings(nn.Module):
    def __init__(self, vocab_size, embed_dim, device="cpu"):
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(vocab_size, self.embed_dim)
        self.device = device

    def forward(self, input):
        ContTensor, CatTensor, LabelTensor, DoseTensor, TimeDiffTensor,VTensor, VancoElTensor, PtList, LengList = input
        CatTensor = CatTensor.to(self.device)
        ContTensor = ContTensor.to(self.device)
        LabelTensor = LabelTensor.to(self.device)
        DoseTensor = DoseTensor.to(self.device)
        TimeDiffTensor = TimeDiffTensor.to(self.device)
        VancoClTensor = VancoElTensor.to(self.device)
        catEmb = self.embed(CatTensor)
        outEmb = torch.sum(catEmb, dim=2)
        outEmb = torch.cat((outEmb, ContTensor), dim=2)
        return outEmb, LabelTensor, LengList, DoseTensor, TimeDiffTensor,VTensor, VancoElTensor, PtList


class PK_RNN(nn.Module):
    def __init__(self, vocab_size=829, embed_dim=8, cont_size=40, hidden_size=32,
                 paramdict={'real_vanco_to_feedback': True,
                            'use_v_from_weight': False,
                            'change_regularize': 'square',
                            'regularize_start_v': 1000,
                            'regularize_start_k': 1000,
                            'regularize_change_v': 1000,
                            'regularize_change_k': 1000},
                 device="cpu"
                 ):
        super().__init__()
        self.emb = EHREmbeddings(vocab_size, embed_dim, device)
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(embed_dim+cont_size+1, hidden_size, device=device)
        self.use_v_from_weight = paramdict['use_v_from_weight']
        self.change_regularize = paramdict['change_regularize']
        self.scale_start_v = paramdict['regularize_start_v']
        self.scale_start_k = paramdict['regularize_start_k']
        self.scale_change_v = paramdict['regularize_change_v']
        self.scale_change_k = paramdict['regularize_change_k']
        self.real_vanco_to_feedback = paramdict['real_vanco_to_feedback']
        self.device = device
        self.out = nn.Linear(hidden_size, 2)

    def forward(self, input):
        outEmb, lb, LengList, d, td, v_from_weight, VancoElTensor, PtList = self.emb(input)
        batchsize = outEmb.shape[0]
        mask = lb != 0
        h_0 = torch.zeros((batchsize, self.hidden_size)).to(self.device)
        hidden = [h_0]
        K0 = VancoElTensor[:, 0]
        V0 = v_from_weight[:, 0]
        TotalMassList = [torch.zeros((len(outEmb), 1)).to(self.device)]
        VancoConcnList = [torch.zeros((len(outEmb))).to(self.device)]
        KV = []
        for i in range(outEmb.shape[1]):  # embed [bs, visit, emb_size+n_conts]
            if self.real_vanco_to_feedback:
                if i == 0:  # no data yet
                    vanco_to_feedback = VancoConcnList[-1].detach()
                else:
                    vanco_to_feedback = VancoConcnList[-1].detach() * (1 - mask[:, i - 1].float()) +\
                                        lb[:, i - 1] * mask[:, i - 1].float()
                    # [bs] = [bs] * [bs] + [bs] * [bs]
            else:
                vanco_to_feedback = VancoConcnList[-1].detach()
            Input_plus_Output = torch.cat((outEmb[:, i, :], vanco_to_feedback.reshape(-1, 1).detach()), dim=1)
            # outEmb [bs, in_size], VancoConcnList[-1] [bs, 1] ----> Input_plus_Output [bs, in_size+1]
            h_i = self.cell(Input_plus_Output, hidden[-1])  # h_i.shape: [bs, hidden_size]
            kv = torch.exp(self.out(h_i))
            k = kv[:, 0]
            v = kv[:, 1]
            if self.use_v_from_weight:
                v = v_from_weight[:, i]
                kv = torch.stack((k, v)).permute(1, 0)
            A = torch.exp(-k * td[:, i])  # A.shape = [bs, seq]
            B = 1 / k * (-torch.exp(-k * d[:, i]) + 1) * torch.exp(-k * (td[:, i] - d[:, i]))
            totalmass = (TotalMassList[-1][:, 0] * A + B)
            VancoConcn = (totalmass / (v + 1e-6))
            VancoConcnList.append(VancoConcn)  # [bs, 1]
            TotalMassList.append(totalmass.reshape(-1, 1))
            KV.append(kv)
            hidden.append(h_i)
        output = torch.stack(VancoConcnList[1:]).permute(1, 0)  # [bs, seq_length]
        #  --------------------- EXTRACT K,V FOR PLOTTING --------------------- #
        KV = torch.stack(KV).permute(1, 0, 2)  # [bs, seq, 2]
        K, V = KV[..., 0], KV[..., 1]
        #  --------------------- CALCULATING LOSS --------------------- #
        if self.change_regularize == 'abs':
            kv_change = (torch.abs(torch.sub(K[:, 1:], K[:, :-1])).mean() * self.scale_change_k
                         + torch.abs(torch.sub(V[:, 1:], V[:, :-1])).mean() * self.scale_change_v)
        else:
            kv_change = ((torch.sub(K[:, 1:], K[:, :-1])).pow(2).mean() * self.scale_change_k
                         + (torch.sub(V[:, 1:], V[:, :-1])).pow(2).mean() * self.scale_change_v)
        mse_loss = ((output - lb)*mask).pow(2).sum()/mask.sum()
        loss = (mse_loss  # different between vancomycin level real and prediction
                # feed k and v at the beginning
                + (K0 - K[:, 0]).pow(2).mean() * self.scale_start_k
                + (V0 - V[:, 0]).pow(2).mean() * self.scale_start_v
                # keep k,v constant not change much
                + kv_change
                )
        return K, V, output, loss, mse_loss


class LSTMAttentionVancomycin(nn.Module):
    """
    LSTM 기반 시계열 인코더와 multi-head attention을 활용해
    반코마이신 농도를 예측하는 PyTorch 모듈.
    입력 형식은 PK_RNN과 동일한 9-튜플을 기대한다.
    """

    def __init__(self, vocab_size=829, embed_dim=16, cont_size=40, lstm_hidden=128,
                 lstm_layers=2, attn_heads=4, dropout=0.2, fc_units=(128, 64),
                 device="cpu"):
        super().__init__()
        self.device = device
        self.embedder = EHREmbeddings(vocab_size, embed_dim, device=device)
        fusion_size = embed_dim + cont_size + 4  # dose, timediff, v, vanco_el
        self.feature_proj = nn.Linear(fusion_size, lstm_hidden)
        self.feature_norm = nn.LayerNorm(lstm_hidden)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden,
            num_heads=attn_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_hidden)
        self.attn_dropout = nn.Dropout(dropout)
        mlp_layers = []
        in_dim = lstm_hidden
        for width in fc_units:
            mlp_layers.append(nn.Linear(in_dim, width))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(dropout))
            in_dim = width
        self.mlp_head = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
        self.output_layer = nn.Linear(in_dim, 1)
        self.last_attention = None

    @staticmethod
    def _build_padding_mask(lengths, max_len):
        idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idx >= lengths.unsqueeze(1)

    def _prepare_inputs(self, batch):
        (features, labels, lengths, dose, tdiff,
         v_from_weight, vanco_el, _) = self.embedder(batch)
        extras = torch.stack(
            (dose, tdiff, v_from_weight, vanco_el),
            dim=2,
        )
        fused = torch.cat((features, extras), dim=2)
        lengths = lengths.to(torch.long).to(self.device)
        labels = labels.to(self.device)
        mask = labels != 0
        padding_mask = self._build_padding_mask(lengths, fused.size(1))
        return fused, labels, mask, padding_mask

    def forward(self, batch):
        fused, labels, obs_mask, padding_mask = self._prepare_inputs(batch)
        projected = self.feature_proj(fused)
        projected = self.feature_norm(projected)
        projected = self.input_dropout(projected)
        lstm_out, _ = self.lstm(projected)
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=padding_mask
        )
        attn_out = self.attn_norm(lstm_out + self.attn_dropout(attn_out))
        features = self.mlp_head(attn_out)
        predictions = self.output_layer(features).squeeze(-1)
        valid_mask = (~padding_mask).float()
        combined_mask = obs_mask.float() * valid_mask
        denom = combined_mask.sum().clamp(min=1.0)
        mse_loss = ((predictions - labels) * combined_mask).pow(2).sum() / denom
        self.last_attention = attn_weights
        return predictions, mse_loss


class VTDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.eta1 = nn.Parameter(torch.zeros(1))
        self.eta2 = nn.Parameter(torch.zeros(1))
        self.eta3 = nn.Parameter(torch.zeros(1))
        self.V2 = 48.3
        self.C = [torch.zeros(1)]
        self.D = [torch.zeros(1)]
        self.A = []
        self.optim = SGD(self.parameters(), lr=1e-2)

    def reset(self):
        self.C = [torch.zeros(1)]
        self.D = [torch.zeros(1)]
        self.A = []

    def update(self, dose, tdiff, ccl):
        V1 = 33.1 * torch.exp(self.eta1)
        Q = 6.99 * torch.exp(self.eta3)
        k10 = ccl * 3.96e-2 * torch.exp(self.eta2) / V1
        k12 = Q / V1
        k21 = Q / self.V2
        R = 1000 / V1
        delta = torch.sqrt((k10 + k12 + k21) ** 2 - 4 * k10 * k21)
        lambda1 = (-k10 - k12 - k21 - delta) / 2
        lambda2 = (-k10 - k12 - k21 + delta) / 2
        c1 = -R * k12 / lambda1 / delta
        c2 = R * k12 / lambda2 / delta
        c3 = (lambda1 + k21) / k12
        c4 = (lambda2 + k21) / k12
        if dose == 0:
            self.C.append(self.C[-1] * torch.exp(lambda1 * tdiff))
            self.D.append(self.D[-1] * torch.exp(lambda2 * tdiff))
        else:
            self.C.append(((self.C[-1] + c1) * torch.exp(lambda1 * dose) - c1) * torch.exp(lambda1 * (tdiff - dose)))
            self.D.append(((self.D[-1] + c2) * torch.exp(lambda2 * dose) - c2) * torch.exp(lambda2 * (tdiff - dose)))
        self.A.append(self.C[-1] * c3 + self.D[-1] * c4)

    def predict(self, dose_tensor, tdiff_tensor, ccl_tensor):
        self.reset()
        with torch.no_grad():
            for i in range(len(dose_tensor)):
                self.update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
        return torch.cat(self.A)

    def loss(self, dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor):
        self.reset()
        mask_tensor = vanco_level_tensor != 0
        index = int(mask_tensor.nonzero()[0])
        for i in range(index + 1):
            self.update(dose_tensor[i], tdiff_tensor[i], ccl_tensor[i])
        loss = (self.A[index] - vanco_level_tensor[
            index]) ** 2 / 4 ** 2 + self.eta1 ** 2 / 0.12 + self.eta2 ** 2 / 0.149 + self.eta3 ** 2 / 0.416
        return loss

    def inference(self, dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor):
        for _ in range(100):
            self.optim.zero_grad()
            l = self.loss(dose_tensor, tdiff_tensor, ccl_tensor, vanco_level_tensor)
            l.backward()
            self.optim.step()
