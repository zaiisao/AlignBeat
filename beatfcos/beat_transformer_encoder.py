import torch
from torch import nn
import sys
from beatfcos.DilatedTransformerLayer import DilatedTransformerLayer

class BeatTransformerEncoder(nn.Module):
    """
    Input : (B, T, 128) mel spectrogram
    Output : C1(B, T, d), C2(B, T, d), C3(B, T, d)
    tap at DSA layer 2, 5, 8 (3 groups of 3)

    Beat-Transformer 원본 중 non-demix ablation(code/ablation_models/non_demix_model.py,
    DilatedTransformerModel) 기반. 원본 대비 변경점:
    - demix(5-stem) 대신 non-demix 채택: Beat This(비교 대상)도 non-demix라 입력 정보량을
      통일하고, FCOS 헤드의 기여를 demixing과 분리하기 위함. 그 결과 demix 전용 pretrained
      체크포인트(dmodel=256, nhead=8, instr_attention 포함)는 구조가 달라 로드 불가 → 처음부터 학습.
      (conv1/conv2는 dmodel에 안 걸려 원본과 shape이 같고, dmodel=256/nhead=8로 맞추면
      time_attention_i까지는 키 리네이밍으로 부분 전이도 가능했으나, 체크포인트 크기에 맞춰
      모델을 키우면 Beat This 대비 용량 confound가 생겨 의도적으로 포기함.)
    - out_linear(beat/downbeat 헤드), out_linear_t(tempo 헤드), skip, inference()(attention
      시각화) 전부 제거: beat/downbeat는 FCOS 헤드가 대체할 예정이고 tempo·attention 시각화는
      이번 스코프 밖.
    - Transformer_layers(nn.ModuleDict, 이름 키) → dsa_layers(nn.ModuleList, 인덱스): 원본은
      instr_attention처럼 이름으로 구분해야 할 서브모듈이 있었지만 non-demix에는 없어서 단순화.
      그 인덱스를 아래 tap-out 조건(i==2/5/8)에 그대로 재사용.

    FIXED ISSUE: nhead 기본값이 2였을 때 DilatedTransformerLayer.py의 8-head
    하드코딩 분할(k[:,0:4]/k[:,4:5]/k[:,5:6]/k[:,6:7]x2)과 맞지 않아, 4개의
    skewed(dilated) head가 빈 텐서가 되어 symmetric head만 남는 버그가 있었음.
    기본값을 8로 고쳐서 dilated head가 실제로 동작하게 함 — 이 분할 로직을 쓰는 한
    nhead는 반드시 8이어야 함(다른 값으로 호출하는 코드가 있다면 버그).
    """
    def __init__(self, attn_len=5, dmodel=128, nhead=8, d_hid=512,
                nlayers=9, norm_first=True, dropout=0.1):
        
        super().__init__()
        assert dmodel % nhead == 0

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,3), stride=1, padding=(2,0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 12), stride=1, padding=(0,0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(64, dmodel, kernel_size=(3, 6), stride=1, padding=(1,0))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(p=dropout)

        self.nlayers = nlayers
        # 원본: nn.ModuleDict({f'time_attention_{idx}': ...}) — 이름 기반 접근.
        # non-demix라 instr_attention 서브모듈이 없어 이름 구분이 불필요해짐 → 인덱스 기반으로 단순화.
        self.dsa_layers = nn.ModuleList([
            DilatedTransformerLayer(
                dmodel, nhead, d_hid, dropout,
                Er_provided=False, attn_len=attn_len, norm_first=norm_first
            )
            for _ in range(nlayers)
        ])

    def forward(self, x):
            x = x.unsqueeze(1)

            x = torch.relu(self.maxpool1(self.conv1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.maxpool2(self.conv2(x)))
            x = self.dropout2(x)
            x = torch.relu(self.maxpool3(self.conv3(x)))
            x = self.dropout3(x)

            x = x.transpose(1, 3).squeeze(1).contiguous()  # (B, T, dmodel)

            # 원본: 9개 레이어를 전부 통과한 마지막 출력만 out_linear(beat/downbeat)로 보냄.
            # skip(self-attention 원시 출력)은 tempo 헤드(out_linear_t) 전용이라 여기선 버림(_).
            # FCOS 헤드가 FPN 멀티스케일 입력을 필요로 해서, 마지막 출력 대신 9개 레이어를
            # 3등분한 지점(2/5/8)에서 hidden state를 tap-out함 — 나눠떨어져서 고른 시작점이며
            # 최적이라는 근거는 없음 (비균등 4구간 분할은 후속 ablation으로 계획됨).
            for i, layer in enumerate(self.dsa_layers):
                x, _ = layer(x, layer=i)
                if i == 2: C1 = x
                if i == 5: C2 = x
                if i == 8: C3 = x

            return C1, C2, C3