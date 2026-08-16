#!/usr/bin/env python3
"""
BeatFCOS Architecture Diagram  (한글 지원 버전)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import matplotlib.patheffects as pe

# ── 한글 폰트 설정 ───────────────────────────────────────
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp       = FontProperties(fname=FONT_PATH)
fp_bold  = FontProperties(fname=FONT_PATH, weight='bold')

def T(ax, x, y, text, size=9, bold=False, color='#222222', ha='center', va='center', **kw):
    f = fp_bold if bold else fp
    ax.text(x, y, text, fontsize=size, fontproperties=f,
            ha=ha, va=va, color=color, **kw)

# ── 색상 ─────────────────────────────────────────────────
C_IN   = '#E3F2FD'
C_CONV = '#E8F5E9'
C_DSA  = '#E8EAF6'
C_TAP  = '#FFF8E1'
C_RHO  = '#F3E5F5'
C_FPN  = '#FFFDE7'
C_HEAD = '#FFEBEE'
C_NEW  = '#E8F5E9'
EDGE   = '#78909C'

fig = plt.figure(figsize=(13, 20))
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 13)
ax.set_ylim(0, 20)
ax.axis('off')
ax.set_facecolor('#FAFAFA')

# ── 헬퍼 ─────────────────────────────────────────────────
def rect(ax, x, y, w, h, color, lw=1.2, radius=0.12):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f'round,pad=0.04,rounding_size={radius}',
                       facecolor=color, edgecolor=EDGE, linewidth=lw, zorder=2)
    ax.add_patch(p)

def arrow(ax, x1, y1, x2, y2, color=EDGE, lw=1.4, style='->'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1), zorder=3,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

# ═══════════════════════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════════════════════
T(ax, 6.5, 19.55, 'BeatFCOS  Architecture', size=14, bold=True)
T(ax, 6.5, 19.15, 'DSA Backbone → rho → FPN → Head', size=10, color='#555')

# ═══════════════════════════════════════════════════════════
# 1. INPUT
# ═══════════════════════════════════════════════════════════
rect(ax, 3.5, 18.3, 6.0, 0.65, C_IN)
T(ax, 6.5, 18.73, 'Mel Spectrogram', size=10, bold=True)
T(ax, 6.5, 18.44, '(B,  T,  128)', size=9, color='#555')
arrow(ax, 6.5, 18.3, 6.5, 17.9)

# ═══════════════════════════════════════════════════════════
# 2. CONV PIPELINE
# ═══════════════════════════════════════════════════════════
rect(ax, 2.5, 17.2, 8.0, 0.65, C_CONV)
T(ax, 6.5, 17.63, 'Conv2d × 3  +  MaxPool × 3', size=10, bold=True)
T(ax, 6.5, 17.34, '(B, 1, T, 128)  →  (B, T, dmodel=512)', size=9, color='#555')
arrow(ax, 6.5, 17.2, 6.5, 16.85)

# ═══════════════════════════════════════════════════════════
# 3. DSA LAYERS  (layer 0 ~ 8)
# ═══════════════════════════════════════════════════════════
T(ax, 6.5, 16.72, 'DSA  (Dilated Self-Attention)  ×  9 layers', size=10, bold=True, color='#1A237E')

DSA_X  = 1.5
DSA_W  = 7.5
DSA_H  = 0.44
DSA_GAP= 0.06
DSA_TOP= 16.45

tap_info = {
    2: ('C1', '±325ms  →  local'),
    5: ('C2', '±2.9s   →  medium'),
    8: ('C3', 'covers T  →  global'),
}
tap_y = {}   # y 중심 저장

for i in range(9):
    y0 = DSA_TOP - i * (DSA_H + DSA_GAP)
    is_tap = i in tap_info
    col = C_TAP if is_tap else C_DSA
    lw  = 1.8  if is_tap else 1.1
    rect(ax, DSA_X, y0 - DSA_H, DSA_W, DSA_H, col, lw=lw)

    dil = 2**i
    T(ax, DSA_X + DSA_W/2, y0 - DSA_H/2,
      f'Layer {i}    dilation = {dil}',
      size=9, bold=is_tap)

    if is_tap:
        tap_y[i] = y0 - DSA_H/2

    # 층 간 화살표
    if i < 8:
        arrow(ax, DSA_X + DSA_W/2, y0 - DSA_H,
              DSA_X + DSA_W/2, y0 - DSA_H - DSA_GAP + 0.01,
              lw=1.1)

# ── TAP 라벨 (오른쪽) ────────────────────────────────────
TAP_X  = DSA_X + DSA_W + 0.15
TAP_BW = 3.8
for i, (name, desc) in tap_info.items():
    cy = tap_y[i]
    rect(ax, TAP_X, cy - 0.22, TAP_BW, 0.44, C_TAP, lw=1.5)
    T(ax, TAP_X + TAP_BW/2, cy + 0.04, f'{name}  (dilation={2**i})', size=9, bold=True, color='#E65100')
    T(ax, TAP_X + TAP_BW/2, cy - 0.14, desc, size=8, color='#555')
    # 화살표: DSA → tap box
    ax.annotate('', xy=(TAP_X, cy), xytext=(DSA_X + DSA_W, cy),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3,
                                linestyle='dashed'))

# ═══════════════════════════════════════════════════════════
# 4. RHO  (temporal downsampling)
# ═══════════════════════════════════════════════════════════
BOT_DSA = DSA_TOP - 9 * (DSA_H + DSA_GAP) + DSA_GAP   # DSA 마지막 box 아래
arrow(ax, DSA_X + DSA_W/2, BOT_DSA, DSA_X + DSA_W/2, BOT_DSA - 0.25)

RHO_Y  = BOT_DSA - 0.85
RHO_H  = 0.58
RHO_W  = 2.8
rho_xs = [1.2, 5.0, 8.9]   # 세 box 왼쪽 x

T(ax, 6.5, RHO_Y + RHO_H + 0.12,
  'rho  (temporal downsampling)',
  size=10, bold=True, color='#4A148C')

for rx, (name, desc) in zip(rho_xs,
        [('rho1  Identity', '(B, 512, T)'),
         ('rho2  stride-2',  '(B, 512, T/2)'),
         ('rho3  stride-4',  '(B, 512, T/4)')]):
    rect(ax, rx, RHO_Y, RHO_W, RHO_H, C_RHO)
    T(ax, rx + RHO_W/2, RHO_Y + RHO_H*0.65, name, size=9, bold=True)
    T(ax, rx + RHO_W/2, RHO_Y + RHO_H*0.25, desc, size=8, color='#555')

# C1/C2/C3 → rho 연결
for tap_i, rx in zip([2, 5, 8], rho_xs):
    cy = tap_y[tap_i]
    bx = rx + RHO_W/2
    # tap_y에서 rho box 위로 수직 화살표
    ax.plot([TAP_X + TAP_BW/2, TAP_X + TAP_BW/2, bx, bx],
            [cy, RHO_Y + RHO_H + 0.55, RHO_Y + RHO_H + 0.55, RHO_Y + RHO_H],
            color='#E65100', lw=1.2, linestyle='--', zorder=1)
    ax.annotate('', xy=(bx, RHO_Y + RHO_H),
                xytext=(bx, RHO_Y + RHO_H + 0.01),
                arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.2))

# ═══════════════════════════════════════════════════════════
# 5. FPN
# ═══════════════════════════════════════════════════════════
FPN_Y = RHO_Y - 1.1
FPN_H = 0.58
FPN_W = 2.8

T(ax, 6.5, FPN_Y + FPN_H + 0.18,
  'FPN  (Feature Pyramid Network, top-down, 256ch)',
  size=10, bold=True, color='#F57F17')

fpn_data = [
    (rho_xs[0], 'P1  (finest)',   '(B, 256, T)'),
    (rho_xs[1], 'P2',             '(B, 256, T/2)'),
    (rho_xs[2], 'P3  (coarsest)', '(B, 256, T/4)'),
]
for rx, name, desc in fpn_data:
    rect(ax, rx, FPN_Y, FPN_W, FPN_H, C_FPN, lw=1.5)
    T(ax, rx + FPN_W/2, FPN_Y + FPN_H*0.65, name, size=9, bold=True, color='#5D4037')
    T(ax, rx + FPN_W/2, FPN_Y + FPN_H*0.25, desc, size=8, color='#555')

# rho → FPN 수직 화살표
for rx in rho_xs:
    arrow(ax, rx + RHO_W/2, RHO_Y, rx + RHO_W/2, FPN_Y + FPN_H, color='#666')

# top-down: P3→P2→P1 수평 화살표
cx_p3 = rho_xs[2] + FPN_W/2
cx_p2 = rho_xs[1] + FPN_W/2
cx_p1 = rho_xs[0] + FPN_W/2
MID_Y = FPN_Y + FPN_H/2

ax.annotate('', xy=(cx_p2 + FPN_W/2 + 0.05, MID_Y),
            xytext=(cx_p3 - 0.05, MID_Y),
            arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.5))
T(ax, (cx_p3 + cx_p2)/2 + 0.3, MID_Y + 0.15, 'up×2', size=7, color='#F57F17')

ax.annotate('', xy=(cx_p1 + FPN_W/2 + 0.05, MID_Y),
            xytext=(cx_p2 - 0.05, MID_Y),
            arrowprops=dict(arrowstyle='->', color='#F57F17', lw=1.5))
T(ax, (cx_p2 + cx_p1)/2 + 0.3, MID_Y + 0.15, 'up×2', size=7, color='#F57F17')

# ═══════════════════════════════════════════════════════════
# 6. FCOS HEADS  (현재, 교체 대상)
# ═══════════════════════════════════════════════════════════
HEAD_Y = FPN_Y - 1.35
HEAD_H = 0.58
HEAD_W = 2.8

# 배경 박스 (교체 대상 표시)
bg = FancyBboxPatch((0.9, HEAD_Y - 0.15), 11.2, HEAD_H + 0.65,
                    boxstyle='round,pad=0.05',
                    facecolor='#FFF3F3', edgecolor='#E53935',
                    linewidth=1.5, linestyle='--', zorder=1)
ax.add_patch(bg)
T(ax, 6.5, HEAD_Y + HEAD_H + 0.35,
  '⚠  현재 FCOS Head  (교체 대상)',
  size=10, bold=True, color='#C62828')

head_data = [
    (rho_xs[0], 'Classification Head', 'beat / downbeat\n(B, L, 2)'),
    (rho_xs[1], 'Regression Head',     'l,  r\n(B, L, 2)'),
    (rho_xs[2], 'Leftness Head',       'quality score\n(B, L, 1)'),
]
for rx, name, desc in head_data:
    rect(ax, rx, HEAD_Y, HEAD_W, HEAD_H, C_HEAD, lw=1.3)
    T(ax, rx + HEAD_W/2, HEAD_Y + HEAD_H*0.68, name, size=8, bold=True)
    T(ax, rx + HEAD_W/2, HEAD_Y + HEAD_H*0.28, desc, size=7.5, color='#555')

for rx in rho_xs:
    arrow(ax, rx + FPN_W/2, FPN_Y, rx + HEAD_W/2, HEAD_Y + HEAD_H)

# Soft-NMS
NMS_Y = HEAD_Y - 0.75
rect(ax, 3.5, NMS_Y, 6.0, 0.52, C_HEAD, lw=1.3)
T(ax, 6.5, NMS_Y + 0.34, 'Soft-NMS', size=10, bold=True, color='#B71C1C')
T(ax, 6.5, NMS_Y + 0.13, '후처리로 중복 prediction 제거', size=8.5, color='#666')
for rx in rho_xs:
    arrow(ax, rx + HEAD_W/2, HEAD_Y, 6.5, NMS_Y + 0.52, color='#999', lw=1.0)

# ═══════════════════════════════════════════════════════════
# 7. TARGET: E2E NMS-free
# ═══════════════════════════════════════════════════════════
NEW_Y = NMS_Y - 1.4
NEW_H = 1.5

bg2 = FancyBboxPatch((0.9, NEW_Y - 0.1), 11.2, NEW_H + 0.55,
                     boxstyle='round,pad=0.05',
                     facecolor='#F1F8E9', edgecolor='#43A047',
                     linewidth=1.8, zorder=1)
ax.add_patch(bg2)

T(ax, 6.5, NEW_Y + NEW_H + 0.3,
  '→  Target: E2E  NMS-free  구조',
  size=10, bold=True, color='#1B5E20')

rect(ax, 1.5, NEW_Y + 0.75, 10.0, 0.58, '#C8E6C9', lw=1.2)
T(ax, 6.5, NEW_Y + 1.1,  'N개  learnable beat queries', size=9, bold=True)
T(ax, 6.5, NEW_Y + 0.88, 'P1/P2/P3  →  key / value    |    queries  →  query  →  cross-attention', size=8.5, color='#333')

rect(ax, 1.5, NEW_Y + 0.1, 10.0, 0.58, '#C8E6C9', lw=1.2)
T(ax, 6.5, NEW_Y + 0.44, 'FFN Prediction Head', size=9, bold=True)
T(ax, 6.5, NEW_Y + 0.22, 'beat / downbeat class  +  beat 시간  직접 예측', size=8.5, color='#333')

rect(ax, 1.5, NEW_Y - 0.55, 10.0, 0.58, '#C8E6C9', lw=1.2)
T(ax, 6.5, NEW_Y - 0.22, 'Hungarian Matching Loss', size=9, bold=True)
T(ax, 6.5, NEW_Y - 0.43, 'NMS 없음  /  Anchor 없음  /  Leftness 없음', size=8.5, color='#333')

arrow(ax, 6.5, NMS_Y, 6.5, NEW_Y + NEW_H + 0.05, color='#1B5E20', lw=1.5)
arrow(ax, 6.5, NEW_Y + 0.75, 6.5, NEW_Y + 0.68, color='#2E7D32', lw=1.2)
arrow(ax, 6.5, NEW_Y + 0.1,  6.5, NEW_Y + 0.03, color='#2E7D32', lw=1.2)

# ═══════════════════════════════════════════════════════════
# 범례
# ═══════════════════════════════════════════════════════════
legend = [
    mpatches.Patch(fc=C_CONV,  ec=EDGE, label='Conv Pipeline'),
    mpatches.Patch(fc=C_DSA,   ec=EDGE, label='DSA Layer'),
    mpatches.Patch(fc=C_TAP,   ec=EDGE, label='Feature Tap (C1/C2/C3)'),
    mpatches.Patch(fc=C_RHO,   ec=EDGE, label='rho Downsampling'),
    mpatches.Patch(fc=C_FPN,   ec=EDGE, label='FPN'),
    mpatches.Patch(fc=C_HEAD,  ec=EDGE, label='FCOS Head (교체 대상)'),
    mpatches.Patch(fc='#C8E6C9', ec='#43A047', label='E2E NMS-free (Target)'),
]
leg = ax.legend(handles=legend, loc='lower left', fontsize=7.5,
                framealpha=0.95, ncol=1, prop=fp,
                bbox_to_anchor=(0.01, 0.01))

out = '/disk1/taegum/mnt/BeatFCOS/arch_diagram.png'
plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='#FAFAFA')
print(f'저장됨: {out}')
