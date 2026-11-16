#!/usr/bin/env python3
"""Make a faceless, narrated YouTube Short from a JSON script.

usage: python make_short.py scripts/great_emu_war.json out.mp4 [--scale 2]

--scale 2 (default) renders 4K 2160x3840; --scale 1 renders 1080x1920.

Everything runs offline: the voice is espeak-ng (with an MBROLA voice when
installed), the visuals are drawn with Pillow and the video is encoded with ffmpeg.
Each scene in the JSON has: say (narration), kicker, big, sub and visual.
"""
import json, math, os, random, re, subprocess, sys, tempfile
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1080, 1920, 30, 44100
GAP = 0.35                       # pause after each scene, seconds
# W x H is the 1080x1920 design space the visuals are drawn in. FW x FH is the output frame:
# the same for vertical Shorts, 1920x1080 for the "wide" layout (visual left, headline right).
FW, FH = W, H
LAYOUT = dict(vox=0, voy=0, tx=W / 2, ky=190, by=390, sy=560, cx=W / 2, cy=1470, cw=930, cn=4, cap=92,
              glow=(0.5, 0.53, 0.48, 0.2), grain=7.0, crf=18)

def set_layout(name, spec):
    global FW, FH
    if name == 'wide':
        FW, FH = 1920, 1080
        LAYOUT.update(vox=30, voy=-490, tx=1510, ky=250, by=440, sy=600, cx=960, cy=975, cw=1500, cn=7, cap=74,
                      glow=(0.3, 0.5, 0.3, 0.42))
    LAYOUT['grain'] = spec.get('grain', LAYOUT['grain']); LAYOUT['crf'] = spec.get('crf', LAYOUT['crf'])
    LAYOUT['maxrate'] = spec.get('maxrate')        # e.g. "5800k" caps the video bitrate (and so the file size)
HERE = os.path.dirname(os.path.abspath(__file__))
BG0, BG1 = (27, 22, 17), (48, 37, 26)
CREAM, GOLD, RED, INK = (243, 230, 200), (242, 177, 52), (217, 72, 59), (20, 16, 12)
EMU_C, EMU_D = (92, 74, 58), (60, 47, 37)
LINE, LIQ, STEEL = (90, 72, 52), (176, 104, 34), (120, 138, 158)   # ground line, liquid, metal
GLOW = None                                                         # optional soft light behind the visuals
POP = [(255, 71, 133), (71, 214, 255), (255, 160, 60), (140, 255, 170), (190, 130, 255)]   # bright colours for figures

def set_palette(pal):
    """Override colours from the script's "palette": bg0, bg1, text, headline, accent, ink, line, liquid, steel."""
    global BG0, BG1, CREAM, GOLD, RED, INK, LINE, LIQ, STEEL, GLOW, POP
    t = lambda k, d: tuple(pal.get(k, d))
    BG0, BG1, CREAM, GOLD, RED = t('bg0', BG0), t('bg1', BG1), t('text', CREAM), t('headline', GOLD), t('accent', RED)
    INK, LINE, LIQ, STEEL = t('ink', INK), t('line', LINE), t('liquid', LIQ), t('steel', STEEL)
    GLOW = tuple(pal['glow']) if 'glow' in pal else None
    POP = [tuple(c) for c in pal.get('pop', POP)]

S = 2   # render scale: layout is in 1080x1920 units, output is (1080*S)x(1920*S); 2 = 4K

class SD:
    """ImageDraw wrapper: takes coordinates in 1080x1920 layout units, draws at S times the size."""
    def __init__(self, d, ox=0, oy=0): self.d, self.ox, self.oy = d, ox, oy
    def p(self, xy):
        ox, oy = self.ox, self.oy
        if xy and isinstance(xy[0], (tuple, list)): return [((v[0] + ox) * S, (v[1] + oy) * S) for v in xy]
        return [(v + (oy if i % 2 else ox)) * S for i, v in enumerate(xy)]
    @staticmethod
    def w(width): return max(1, round(width * S))
    def line(self, xy, fill=None, width=1, joint=None): self.d.line(self.p(xy), fill=fill, width=self.w(width), joint=joint)
    def ellipse(self, xy, fill=None, outline=None, width=1): self.d.ellipse(self.p(xy), fill=fill, outline=outline, width=self.w(width))
    def rectangle(self, xy, fill=None, outline=None, width=1): self.d.rectangle(self.p(xy), fill=fill, outline=outline, width=self.w(width))
    def polygon(self, xy, fill=None): self.d.polygon(self.p(xy), fill=fill)
    def pieslice(self, xy, a, b, fill=None): self.d.pieslice(self.p(xy), a, b, fill=fill)
    def arc(self, xy, a, b, fill=None, width=1): self.d.arc(self.p(xy), a, b, fill=fill, width=self.w(width))
    def rounded_rectangle(self, xy, radius=0, fill=None, outline=None, width=1):
        self.d.rounded_rectangle(self.p(xy), radius * S, fill=fill, outline=outline, width=self.w(width))
    def text(self, xy, s, font=None, fill=None, anchor=None, stroke_width=0, stroke_fill=None):
        self.d.text(((xy[0] + self.ox) * S, (xy[1] + self.oy) * S), s, font=font, fill=fill, anchor=anchor, stroke_width=round(stroke_width * S), stroke_fill=stroke_fill)
    def textlength(self, s, font=None): return self.d.textlength(s, font=font) / S

_fonts = {}
def font(name, size):
    key = (name, int(size), S)
    if key not in _fonts:
        path = {'anton': 'fonts/Anton-Regular.ttf', 'inter': 'fonts/Inter.ttf'}[name]
        f = ImageFont.truetype(os.path.join(HERE, path), max(1, int(size * S)))
        if name == 'inter':
            try: f.set_variation_by_axes([700])
            except Exception: pass
        _fonts[key] = f
    return _fonts[key]

def ease(t): t = min(max(t, 0.0), 1.0); return 1 - (1 - t) ** 3
def back(t):
    if t <= 0: return 0.0
    t = min(t, 1.0); c = 1.70158
    return 1 + (c + 1) * (t - 1) ** 3 + c * (t - 1) ** 2

# ---------- audio ----------
def load_wav(path):
    raw = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    nz = np.where(np.abs(a) > 0.01)[0]          # trim silence at both ends
    return a[max(nz[0] - 800, 0): nz[-1] + 1600] if len(nz) else a

_kokoro = None
def tts(text, voice, speed, path):
    """voice "kokoro:<name>" uses the Kokoro neural voice (natural sounding, speed is a
    multiplier like 1.0); any other voice name uses espeak-ng (speed in words per minute)."""
    global _kokoro
    if voice.startswith('kokoro:'):
        if _kokoro is None:
            from kokoro_onnx import Kokoro
            d = os.environ.get('KOKORO_DIR', '/root/tts')
            _kokoro = Kokoro(os.path.join(d, 'kokoro-v1.0.onnx'), os.path.join(d, 'voices-v1.0.bin'))
        import soundfile as sf
        a, sr = _kokoro.create(text, voice=voice.split(':', 1)[1], speed=float(speed), lang='en-us')
        sf.write(path, a, sr)
        return
    # -g adds a short gap between words, which makes the synthetic voice easier to follow
    try:
        subprocess.run(['espeak-ng', '-v', voice, '-s', str(speed), '-g', '3', '-w', path, text], check=True, capture_output=True)
    except subprocess.CalledProcessError:   # MBROLA voice missing: fall back to the plain voice
        subprocess.run(['espeak-ng', '-v', 'en-us', '-s', str(speed), '-g', '3', '-w', path, text], check=True)

# speech clean-up: the robotic voice needs strong EQ; the neural voice only a gentle polish
VOICE_FX_ROBOT = ('highpass=f=100,equalizer=f=250:t=q:w=1:g=-3,equalizer=f=2800:t=q:w=1.2:g=5,'
                  'equalizer=f=5500:t=q:w=1:g=3,acompressor=threshold=-20dB:ratio=3:attack=5:release=80:makeup=2')
VOICE_FX_NEURAL = ('highpass=f=80,equalizer=f=3500:t=q:w=1.5:g=1.5,'
                   'acompressor=threshold=-18dB:ratio=2:attack=10:release=150:makeup=1.5')

def envelope(a, win):
    """Moving average of |a| over win samples (fast, via cumulative sum)."""
    c = np.concatenate([[0], np.cumsum(np.abs(a), dtype=np.float64)])
    h = win // 2; i = np.arange(len(a))
    lo, hi = np.clip(i - h, 0, len(a)), np.clip(i + h, 0, len(a))
    return ((c[hi] - c[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)

def music_bed(n):
    t = np.arange(n) / SR
    out = np.zeros(n, np.float32)
    for f, a in [(55, .5), (82.4, .35), (110, .3), (130.8, .18), (164.8, .12)]:
        lfo = 0.6 + 0.4 * np.sin(2 * math.pi * (0.07 + f / 4000) * t + f)
        out += (a * lfo * np.sin(2 * math.pi * f * t)).astype(np.float32)
    pulse = (np.sin(2 * math.pi * 1.0 * t) > 0.97).astype(np.float32)          # soft heartbeat
    kick = np.convolve(pulse, np.exp(-np.arange(4000) / 600) * np.sin(2 * math.pi * 50 * np.arange(4000) / SR), 'same')
    out = out / np.abs(out).max() * 0.035 + kick.astype(np.float32) * 0.02
    fade = np.minimum(1, np.minimum(t / 1.5, (t[-1] - t) / 1.5))
    return out * fade

def whoosh(n_len=int(0.35 * SR)):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(n_len).astype(np.float32)
    x = np.convolve(x, np.ones(30) / 30, 'same')
    env = np.sin(np.linspace(0, math.pi, n_len)) ** 2
    return x * env * 0.08

# ---------- drawing helpers ----------
def base_frames(k=6):
    """Background gradient + vignette + film grain, pre-rendered at full output size."""
    RW, RH = FW * S, FH * S
    y = np.linspace(0, 1, RH)[:, None, None]
    img = (np.array(BG0) * (1 - y) + np.array(BG1) * y) * np.ones((1, RW, 1))
    yy, xx = np.mgrid[0:RH, 0:RW]
    v = 1 - 0.55 * (((xx - RW / 2) / (RW * 0.75)) ** 2 + ((yy - RH / 2) / (RH * 0.62)) ** 2)
    img = (img * np.clip(v, 0.25, 1)[:, :, None]).astype(np.float32)
    if GLOW:                                    # soft coloured light behind the visual area
        gx, gy, gw, gh = LAYOUT['glow']
        gl = np.exp(-(((xx - RW * gx) / (RW * gw)) ** 2 + ((yy - RH * gy) / (RH * gh)) ** 2))
        img += (np.array(GLOW, np.float32) * 0.42) * gl[:, :, None].astype(np.float32)
    rng = np.random.default_rng(1); g = 2 * S
    out = []
    for _ in range(k):
        n = rng.normal(0, LAYOUT['grain'], (RH // g, RW // g, 1)).astype(np.float32).repeat(g, 0).repeat(g, 1)
        out.append(Image.fromarray(np.clip(img + n, 0, 255).astype(np.uint8)).convert('RGBA'))
    return out

def text_c(d, xy, s, f, fill, anchor='mm', stroke=0, sfill=None):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sfill or INK)

def emu(d, x, y, s, phase=0.0, face=1, col=EMU_C, run=False):
    """A simple emu silhouette standing on (x, y), about 300*s px tall."""
    f = face
    bob = math.sin(phase * 2 * math.pi) * 6 * s * (1.6 if run else 1)
    by = y - 150 * s + bob
    # legs
    for k in (0, 0.5):
        a = math.sin((phase + k) * 2 * math.pi) * (0.55 if run else 0.25)
        hx, hy = x, by + 30 * s
        kx, ky = hx + math.sin(a) * 60 * s, hy + math.cos(a) * 60 * s
        fx, fy = kx - f * 18 * s + math.sin(a * 0.6) * 25 * s, y
        d.line([(hx, hy), (kx, ky), (fx, fy)], fill=EMU_D, width=max(2, int(9 * s)), joint='curve')
        d.line([(fx, fy), (fx + f * 22 * s, fy)], fill=EMU_D, width=max(2, int(6 * s)))
    # body + tail feathers
    d.ellipse([x - 95 * s, by - 55 * s, x + 95 * s, by + 50 * s], fill=col)
    for i in range(5):
        tx = x - f * (80 + i * 6) * s
        d.line([(tx, by - 30 * s + i * 14 * s), (tx - f * 35 * s, by - 10 * s + i * 16 * s)], fill=EMU_D, width=max(2, int(7 * s)))
    # neck and head
    nb = (x + f * 70 * s, by - 25 * s)
    nm = (x + f * 95 * s, by - 95 * s)
    hd = (x + f * 88 * s, by - 160 * s + bob * 0.3)
    d.line([nb, nm, hd], fill=col, width=max(3, int(22 * s)), joint='curve')
    d.ellipse([hd[0] - 20 * s, hd[1] - 18 * s, hd[0] + 20 * s, hd[1] + 18 * s], fill=EMU_D)
    d.polygon([(hd[0] + f * 14 * s, hd[1] - 6 * s), (hd[0] + f * 46 * s, hd[1] + 2 * s), (hd[0] + f * 14 * s, hd[1] + 8 * s)], fill=(200, 160, 110))
    d.ellipse([hd[0] + f * 4 * s - 5 * s, hd[1] - 8 * s, hd[0] + f * 4 * s + 5 * s, hd[1] + 2 * s], fill=CREAM)

def bullet(d, x, y, s, col=GOLD):
    d.rounded_rectangle([x - 10 * s, y - 10 * s, x + 10 * s, y + 34 * s], 3 * s, fill=(184, 115, 51))
    d.pieslice([x - 10 * s, y - 32 * s, x + 10 * s, y + 12 * s], 180, 360, fill=col)
    d.rectangle([x - 10 * s, y - 11 * s, x + 10 * s, y - 9 * s], fill=col)

def gun(d, x, y, s, col=(70, 70, 70)):
    d.rounded_rectangle([x - 200 * s, y - 30 * s, x + 120 * s, y + 30 * s], 10 * s, fill=col)    # body
    d.rectangle([x + 120 * s, y - 12 * s, x + 330 * s, y + 12 * s], fill=col)                   # barrel
    d.ellipse([x - 60 * s, y - 110 * s, x + 60 * s, y - 50 * s], fill=(90, 90, 90))            # pan magazine
    d.line([(x - 40 * s, y + 30 * s), (x - 120 * s, y + 160 * s)], fill=col, width=int(16 * s))  # legs
    d.line([(x + 60 * s, y + 30 * s), (x + 140 * s, y + 160 * s)], fill=col, width=int(16 * s))
    d.polygon([(x - 200 * s, y - 20 * s), (x - 320 * s, y + 10 * s), (x - 320 * s, y + 50 * s), (x - 200 * s, y + 30 * s)], fill=(110, 80, 50))

def shade(c, k): return tuple(max(0, min(255, int(v * k))) for v in c)

def tank(d, x, ground, w, h, level=1.0, shake=0.0, crack=0.0):
    """A riveted storage tank standing on the ground; level 0..1 is how full it is."""
    x += shake
    x0, x1, y0 = x - w / 2, x + w / 2, ground - h
    d.ellipse([x0, y0 - 40, x1, y0 + 40], fill=shade(STEEL, 1.15))
    d.rectangle([x0, y0, x1, ground], fill=STEEL)
    ly = ground - (h - 20) * level
    d.rectangle([x0 + 14, ly, x1 - 14, ground - 8], fill=LIQ)
    for i in range(1, 5):
        yy = y0 + h * i / 5
        d.line([(x0, yy), (x1, yy)], fill=shade(STEEL, 0.7), width=5)
        for k in range(9):
            rx = x0 + 20 + k * (w - 40) / 8
            d.ellipse([rx - 4, yy - 16, rx + 4, yy - 8], fill=shade(STEEL, 0.6))
    if crack > 0:
        rng = random.Random(5); pts = [(x + 20, y0 + 30)]
        for i in range(int(9 * crack)):
            px, py = pts[-1]; pts.append((px + rng.uniform(-45, 45), py + h / 10))
        if len(pts) > 1:
            d.line(pts, fill=INK, width=9, joint='curve'); d.line(pts, fill=LIQ, width=4, joint='curve')

def buildings(d, x0, ground, n=5, seed=2, scale=1.0, tilt=0.0):
    rng = random.Random(seed); x = x0
    for i in range(n):
        bw, bh = rng.uniform(90, 140) * scale, rng.uniform(170, 330) * scale
        col = shade(STEEL, rng.uniform(0.35, 0.55)); lean = tilt * (i + 1) * 9
        d.polygon([(x, ground), (x + lean, ground - bh), (x + bw + lean, ground - bh), (x + bw, ground)], fill=col)
        for r in range(int(bh // 60)):
            for c in range(int(bw // 40)):
                wx, wy = x + 14 + c * 40 + lean * (r * 60 + 40) / bh, ground - bh + 22 + r * 60
                d.rectangle([wx, wy, wx + 16, wy + 26], fill=shade(GOLD, 0.85) if rng.random() > 0.35 else shade(col, 0.6))
        x += bw + 10 * scale

def wave(d, front, ground, height, t, left=-100):
    """A wall of liquid from the left edge up to x=front, with a curling crest."""
    pts = [(left, ground)]
    for i in range(41):
        u = i / 40; x = left + (front - left) * u
        y = ground - height * (0.45 + 0.55 * u ** 1.6) + math.sin(u * 9 + t * 5) * 14
        pts.append((x, y))
    pts += [(front + 70, ground - height * 0.55), (front + 40, ground - height * 0.15), (front + 90, ground)]
    d.polygon(pts, fill=LIQ)
    d.line(pts[1:42], fill=shade(LIQ, 1.45), width=10, joint='curve')
    for i in range(6):
        bx = left + (front - left) * ((i * 0.17 + t * 0.2) % 1); by = ground - height * 0.2 - (i * 37 % 90)
        d.ellipse([bx - 12, by - 12, bx + 12, by + 12], fill=shade(LIQ, 1.3))

def person(d, x, y, s, col):
    d.ellipse([x - 13 * s, y - 62 * s, x + 13 * s, y - 36 * s], fill=col)
    d.rounded_rectangle([x - 18 * s, y - 32 * s, x + 18 * s, y + 22 * s], 12 * s, fill=col)

def mix(a, b, k): return tuple(int(a[i] * (1 - k) + b[i] * k) for i in range(3))

def dancer(d, x, y, s, phase, col, style=0):
    """A dancing figure with feet near (x, y); phase animates the limbs."""
    w = max(2, int(13 * s)); a = phase * 2 * math.pi
    hop = abs(math.sin(a)) * 26 * s; sway = math.sin(a) * 16 * s
    hip = (x + sway * 0.4, y - 120 * s - hop); sh = (x + sway, y - 235 * s - hop)
    for k, sg in ((0, -1), (math.pi, 1)):                                   # legs
        kick = math.sin(a + k) * 0.5 + 0.5
        knee = (hip[0] + sg * (22 + 30 * kick) * s, hip[1] + (62 - 14 * kick) * s)
        foot = (knee[0] + sg * (8 + 34 * kick) * s, y - 46 * kick * s - hop * 0.2)
        d.line([hip, knee, foot], fill=col, width=w, joint='curve')
    if style % 2:                                                           # a skirt on every other dancer
        d.polygon([(sh[0] - 22 * s, sh[1] + 60 * s), (hip[0] - 62 * s, hip[1] + 46 * s), (hip[0] + 62 * s, hip[1] + 46 * s), (sh[0] + 22 * s, sh[1] + 60 * s)], fill=col)
    d.line([hip, sh], fill=col, width=int(w * 2.1))
    for sg, off in ((-1, 0.0), (1, 1.4)):                                   # arms
        up = math.sin(a * 2 + off + style)
        el = (sh[0] + sg * 52 * s, sh[1] - (10 + 34 * up) * s)
        hand = (el[0] + sg * (22 + 14 * up) * s, el[1] - (48 + 22 * up) * s)
        d.line([(sh[0], sh[1] + 8 * s), el, hand], fill=col, width=w, joint='curve')
    hx, hy = sh[0] + sway * 0.3, sh[1] - 44 * s
    d.ellipse([hx - 27 * s, hy - 27 * s, hx + 27 * s, hy + 27 * s], fill=col)

def note(d, x, y, s, col):
    d.ellipse([x - 16 * s, y - 11 * s, x + 12 * s, y + 13 * s], fill=col)
    d.line([(x + 9 * s, y), (x + 9 * s, y - 74 * s)], fill=col, width=max(2, int(7 * s)))
    d.line([(x + 9 * s, y - 72 * s), (x + 44 * s, y - 54 * s)], fill=col, width=max(2, int(11 * s)))

def notes(d, t, x0, x1, y0, y1, n=7):
    rng = random.Random(11)
    for i in range(n):
        u = (t * rng.uniform(0.18, 0.32) + rng.random()) % 1
        x = x0 + (x1 - x0) * rng.random() + math.sin(t * 2 + i) * 22
        note(d, x, y1 - (y1 - y0) * u, rng.uniform(0.6, 1.0), mix(POP[i % len(POP)], BG1, 0.55 * u))

def sparkles(d, t, n=26):
    """Slow drifting dots of light across the visual area."""
    rng = random.Random(21)
    for i in range(n):
        x = rng.uniform(40, W - 40) + math.sin(t * 0.8 + i) * 18
        y = VY1 + 60 - ((t * rng.uniform(22, 60) + rng.uniform(0, 800)) % 800)
        r = rng.uniform(3, 9); tw = 0.5 + 0.5 * math.sin(t * 3 + i * 1.7)
        c = mix(BG1, POP[i % len(POP)], 0.25 + 0.45 * tw)
        d.ellipse([x - r, y - r, x + r, y + r], fill=c)

# ----- drawings for the long-form "wide" stories -----
FIRE, HOT, ASH, SNOW, LEAF = (255, 120, 30), (255, 214, 90), (86, 80, 90), (236, 244, 250), (92, 150, 90)

def volcano(d, cx, ground, w=760, h=500, top=0.13, shake=0.0, crater=False):
    cx += shake; col = STEEL; ty = ground - h
    d.polygon([(cx - w / 2, ground), (cx - w * top, ty), (cx + w * top, ty), (cx + w / 2, ground)], fill=col)
    d.polygon([(cx - w * top, ty), (cx - w * 0.03, ty + h * 0.42), (cx + w * top * 0.5, ty)], fill=shade(col, 0.78))
    d.polygon([(cx + w * top, ty), (cx + w * 0.2, ty + h * 0.5), (cx + w * 0.33, ty + h * 0.52), (cx + w / 2, ground), (cx + w * 0.3, ground)], fill=shade(col, 0.86))
    if crater: d.ellipse([cx - w * top, ty - 16, cx + w * top, ty + 22], fill=shade(col, 0.5))
    return ty

def plume(d, cx, y0, t, power=1.0, n=34):
    rng = random.Random(4); puffs = []
    for i in range(n):
        age = (t * 0.33 + i / n) % 1
        x = cx + math.sin(i * 12.9 + t * 0.6) * age * 240 * power + rng.uniform(-30, 30) * age
        puffs.append((age, x, y0 - age * 540 * power, (26 + age * 115) * power))
    for age, x, y, r in sorted(puffs, reverse=True):
        col = mix(HOT, FIRE, min(1, age * 5)) if age < 0.2 else mix(FIRE, ASH, min(1, (age - 0.2) * 2.4))
        d.ellipse([x - r, y - r, x + r, y + r], fill=col)

def rocks(d, cx, y0, t, n=14, power=1.0):
    rng = random.Random(8)
    for i in range(n):
        ph = (t * 0.7 + i / n) % 1; vx = rng.uniform(-420, 420) * power
        x, y = cx + vx * ph, y0 - (900 * ph - 1000 * ph * ph) * power; r = rng.uniform(6, 13)
        d.ellipse([x - r, y - r, x + r, y + r], fill=mix(HOT, FIRE, ph))

def lava(d, cx, ty, ground, w, t):
    for i, (dx, ln) in enumerate(((-0.07, 0.75), (0.05, 0.95), (0.11, 0.6), (-0.02, 0.5))):
        k = ease(t / 2.5 - i * 0.15) * ln; x0 = cx + dx * w
        pts = [(x0 + math.sin(j * 1.3 + i) * 12 + dx * w * j * 0.45, ty + (ground - ty) * k * j / 6) for j in range(7)]
        d.line(pts, fill=FIRE, width=16, joint='curve'); d.line(pts, fill=HOT, width=6, joint='curve')

def sea(d, ground, h=40): d.rectangle([0, ground, W, ground + h], fill=LIQ)

def flake(d, x, y, r, col):
    for a in (0, 60, 120):
        dx, dy = math.cos(math.radians(a)) * r, math.sin(math.radians(a)) * r
        d.line([(x - dx, y - dy), (x + dx, y + dy)], fill=col, width=3)

def snowfall(d, t, n=60, col=None, dots=False, top=None, bottom=None):
    rng = random.Random(31); top = VY0 - 20 if top is None else top; bottom = VY1 if bottom is None else bottom
    for i in range(n):
        x = (rng.uniform(0, W) + math.sin(t * 1.3 + i) * 26) % W
        y = top + (t * rng.uniform(70, 150) + rng.uniform(0, bottom - top)) % (bottom - top)
        r = rng.uniform(5, 11)
        if dots: d.ellipse([x - r * 0.6, y - r * 0.6, x + r * 0.6, y + r * 0.6], fill=col or ASH)
        else: flake(d, x, y, r, col or SNOW)

def rain(d, t, n=70, top=None, bottom=None, col=(130, 160, 200)):
    rng = random.Random(41); top = VY0 - 20 if top is None else top; bottom = VY1 - 40 if bottom is None else bottom
    for i in range(n):
        x = rng.uniform(-80, W); y = top + (t * rng.uniform(700, 1000) + rng.uniform(0, 700)) % (bottom - top)
        d.line([(x, y), (x - 14, y + 46)], fill=col, width=3)

def bolt(d, x, y, s, col=None):
    pts = [(x, y), (x - 50 * s, y + 130 * s), (x + 10 * s, y + 130 * s), (x - 60 * s, y + 290 * s), (x + 70 * s, y + 95 * s), (x + 10 * s, y + 95 * s), (x + 50 * s, y)]
    d.polygon(pts, fill=col or HOT)

def house(d, x, ground, w, h, col, roof, lit=True, snow=0.0):
    d.rectangle([x, ground - h, x + w, ground], fill=col)
    d.polygon([(x - 16, ground - h), (x + w / 2, ground - h - w * 0.42), (x + w + 16, ground - h)], fill=roof)
    if snow > 0:
        d.polygon([(x - 16, ground - h), (x + w / 2, ground - h - w * 0.42), (x + w + 16, ground - h), (x + w / 2, ground - h - w * 0.42 + 26 * snow + 14)], fill=SNOW)
    for i in range(max(1, int(w // 90))):
        wx = x + 26 + i * 90
        d.rectangle([wx, ground - h + 34, wx + 40, ground - h + 86], fill=HOT if lit else shade(col, 0.6))
    d.rectangle([x + w / 2 - 20, ground - 70, x + w / 2 + 20, ground], fill=shade(col, 0.55))

def tree(d, x, ground, s, bare=True, col=None):
    col = col or shade(STEEL, 0.6)
    d.line([(x, ground), (x, ground - 150 * s)], fill=col, width=max(3, int(12 * s)))
    if bare:
        for a, ln, yy in ((-50, 80, 90), (50, 70, 110), (-35, 60, 140), (40, 55, 150)):
            d.line([(x, ground - yy * s), (x + math.sin(math.radians(a)) * ln * s, ground - yy * s - math.cos(math.radians(a)) * ln * s)], fill=col, width=max(2, int(7 * s)))
    else:
        d.ellipse([x - 70 * s, ground - 270 * s, x + 70 * s, ground - 120 * s], fill=LEAF)

def cal_card(d, x, y, month, year, col, s=1.0):
    wd, hg = 135 * s, 160 * s
    d.rounded_rectangle([x - wd, y - hg, x + wd, y + hg], 26 * s, fill=CREAM)
    d.rounded_rectangle([x - wd, y - hg, x + wd, y - hg + 96 * s], 26 * s, fill=col)
    d.rectangle([x - wd, y - hg + 60 * s, x + wd, y - hg + 96 * s], fill=col)
    text_c(d, (x, y - hg + 50 * s), str(year), font('inter', 40 * s), INK)
    text_c(d, (x, y + 50 * s), month, font('anton', 130 * s), INK)

def ghost(d, x, y, s, col=None):
    col = col or SNOW
    d.pieslice([x - 40 * s, y - 50 * s, x + 40 * s, y + 30 * s], 180, 360, fill=col)
    d.rectangle([x - 40 * s, y - 10 * s, x + 40 * s, y + 46 * s], fill=col)
    for i in range(3):
        d.ellipse([x - 40 * s + i * 27 * s, y + 32 * s, x - 14 * s + i * 27 * s, y + 58 * s], fill=col)
    for sx in (-14, 14):
        d.ellipse([x + sx * s - 7 * s, y - 18 * s, x + sx * s + 7 * s, y - 2 * s], fill=INK)

def bat(d, x, y, s, flap, col=None):
    col = col or INK; up = math.sin(flap) * 34 * s
    d.ellipse([x - 13 * s, y - 16 * s, x + 13 * s, y + 20 * s], fill=col)
    for sg in (-1, 1):
        d.polygon([(x, y - 6 * s), (x + sg * 50 * s, y - 30 * s - up), (x + sg * 100 * s, y - up * 0.4), (x + sg * 74 * s, y + 10 * s - up * 0.2),
                   (x + sg * 54 * s, y - 2 * s), (x + sg * 34 * s, y + 16 * s), (x + sg * 16 * s, y + 4 * s)], fill=col)
        d.polygon([(x + sg * 4 * s, y - 14 * s), (x + sg * 12 * s, y - 32 * s), (x + sg * 14 * s, y - 12 * s)], fill=col)

def wheel(d, x, y, r, rot, col):
    d.ellipse([x - r, y - r, x + r, y + r], outline=col, width=9)
    for k in range(6):
        a = rot + k * math.pi / 3
        d.line([(x, y), (x + math.cos(a) * r, y + math.sin(a) * r)], fill=col, width=4)

def wagon(d, x, ground, s, t):
    r = 38 * s; by = ground - r - 8 * s
    d.rectangle([x - 90 * s, by - 44 * s, x + 90 * s, by], fill=(132, 92, 58))
    d.pieslice([x - 96 * s, by - 150 * s, x + 96 * s, by + 60 * s], 180, 360, fill=CREAM)
    for wx in (-58, 58): wheel(d, x + wx * s, ground - r, r, t * 4, (70, 50, 36))
    d.line([(x + 90 * s, by - 16 * s), (x + 170 * s, by - 6 * s)], fill=(70, 50, 36), width=max(2, int(7 * s)))

def draisine(d, x, ground, t, col):
    r = 72; wy = ground - r; rot = t * 3.4
    wheel(d, x - 120, wy, r, rot, CREAM); wheel(d, x + 120, wy, r, rot, CREAM)
    d.line([(x - 120, wy), (x - 30, wy - 70), (x + 96, wy - 70), (x + 120, wy)], fill=(150, 104, 62), width=16, joint='curve')
    d.line([(x + 96, wy - 70), (x + 86, wy - 150), (x + 50, wy - 160)], fill=(150, 104, 62), width=10, joint='curve')
    hip = (x - 20, wy - 86); sh = (x + 14, wy - 200)
    for k in (0, math.pi):
        sw = math.sin(t * 6 + k)
        knee = (hip[0] + 26 + sw * 30, hip[1] + 64); foot = (knee[0] - 10 + sw * 44, ground - (12 if sw > 0 else 0) - max(0, -sw) * 30)
        d.line([hip, knee, foot], fill=col, width=15, joint='curve')
    d.line([hip, sh], fill=col, width=28)
    d.line([sh, (x + 46, wy - 176), (x + 60, wy - 156)], fill=col, width=13, joint='curve')
    d.ellipse([sh[0] - 4, sh[1] - 66, sh[0] + 52, sh[1] - 10], fill=col)
    d.rectangle([sh[0] - 8, sh[1] - 92, sh[0] + 56, sh[1] - 62], fill=INK); d.rectangle([sh[0] - 22, sh[1] - 66, sh[0] + 70, sh[1] - 56], fill=INK)   # top hat

# ----- drawings for the "sense of time" facts -----
SAND, STONE = (232, 190, 110), (206, 186, 156)

def pyramid(d, cx, ground, w, h, k=1.0, col=None, n=9):
    """A stepped pyramid built row by row; k 0..1 is how much of it stands."""
    col = col or SAND
    for i in range(n):
        if i / n >= k: break
        y0, y1 = ground - h * i / n, ground - h * (i + 1) / n
        a, b = w / 2 * (1 - i / n), w / 2 * (1 - (i + 1) / n)
        d.polygon([(cx - a, y0), (cx - b, y1), (cx, y1), (cx, y0)], fill=col)
        d.polygon([(cx, y0), (cx, y1), (cx + b, y1), (cx + a, y0)], fill=shade(col, 0.72))
        d.line([(cx - a, y0), (cx + a, y0)], fill=shade(col, 0.55), width=3)

def queen(d, x, y, s):
    """A simple Egyptian queen: gold striped headdress, collar and robe. (x, y) is the feet."""
    hy = y - 250 * s
    d.polygon([(x - 62 * s, hy - 40 * s), (x + 62 * s, hy - 40 * s), (x + 78 * s, hy + 110 * s), (x - 78 * s, hy + 110 * s)], fill=GOLD)
    for i in range(5):
        yy = hy - 26 * s + i * 30 * s
        d.line([(x - (64 + i * 3) * s, yy), (x + (64 + i * 3) * s, yy)], fill=(30, 90, 150), width=max(2, int(9 * s)))
    d.ellipse([x - 42 * s, hy - 34 * s, x + 42 * s, hy + 56 * s], fill=(204, 148, 100))
    d.rounded_rectangle([x - 66 * s, hy + 80 * s, x + 66 * s, y], 26 * s, fill=CREAM)
    d.pieslice([x - 66 * s, hy + 40 * s, x + 66 * s, hy + 150 * s], 0, 180, fill=GOLD)
    d.ellipse([x - 9 * s, hy - 62 * s, x + 9 * s, hy - 36 * s], fill=RED)

def mammoth(d, x, ground, s, t):
    """A woolly mammoth facing right, feet on the ground at x."""
    fur, dark = (128, 82, 50), (92, 58, 36); bob = math.sin(t * 3) * 6 * s
    for i, lx in enumerate((-120, -60, 60, 120)):
        sw = math.sin(t * 3 + i * 1.6) * 16 * s
        d.rounded_rectangle([x + lx * s - 26 * s + sw, ground - 150 * s, x + lx * s + 26 * s + sw, ground], 16 * s, fill=dark if i % 2 else fur)
    d.ellipse([x - 190 * s, ground - 330 * s + bob, x + 150 * s, ground - 90 * s + bob], fill=fur)
    d.ellipse([x + 20 * s, ground - 390 * s + bob, x + 200 * s, ground - 190 * s + bob], fill=fur)          # shoulder hump
    hx, hy = x + 215 * s, ground - 290 * s + bob
    d.ellipse([hx - 80 * s, hy - 90 * s, hx + 70 * s, hy + 80 * s], fill=fur)                              # head
    d.ellipse([hx - 60 * s, hy - 120 * s, hx + 30 * s, hy - 50 * s], fill=fur)
    d.ellipse([hx - 78 * s, hy - 40 * s, hx - 20 * s, hy + 50 * s], fill=dark)                              # ear
    tr = [(hx + 50 * s, hy + 20 * s), (hx + 78 * s, hy + 120 * s), (hx + 70 * s, hy + 210 * s + math.sin(t * 2) * 10 * s), (hx + 110 * s, hy + 240 * s + math.sin(t * 2) * 14 * s)]
    d.line(tr, fill=fur, width=max(4, int(38 * s)), joint='curve')
    d.arc([hx - 10 * s, hy - 40 * s, hx + 230 * s, hy + 170 * s], 20, 165, fill=SNOW, width=max(3, int(17 * s)))   # tusk
    d.ellipse([hx + 14 * s, hy - 26 * s, hx + 30 * s, hy - 10 * s], fill=INK)
    for i in range(9):                                                                                    # shaggy belly
        fx = x - 150 * s + i * 34 * s
        d.polygon([(fx, ground - 120 * s + bob), (fx + 34 * s, ground - 120 * s + bob), (fx + 17 * s, ground - 78 * s + bob)], fill=dark)
    d.line([(x - 185 * s, ground - 230 * s + bob), (x - 215 * s, ground - 150 * s + bob)], fill=dark, width=max(3, int(11 * s)))

def lander(d, x, ground, s):
    d.line([(x - 60 * s, ground - 70 * s), (x - 110 * s, ground)], fill=STEEL, width=max(3, int(9 * s))); d.line([(x + 60 * s, ground - 70 * s), (x + 110 * s, ground)], fill=STEEL, width=max(3, int(9 * s)))
    for fx in (-110, 110): d.ellipse([x + fx * s - 22 * s, ground - 8 * s, x + fx * s + 22 * s, ground + 8 * s], fill=STEEL)
    d.rectangle([x - 78 * s, ground - 130 * s, x + 78 * s, ground - 62 * s], fill=(214, 168, 62))
    d.polygon([(x - 54 * s, ground - 130 * s), (x - 36 * s, ground - 220 * s), (x + 36 * s, ground - 220 * s), (x + 54 * s, ground - 130 * s)], fill=shade(STONE, 0.9))
    d.rectangle([x - 14 * s, ground - 196 * s, x + 14 * s, ground - 160 * s], fill=INK)
    d.line([(x + 30 * s, ground - 220 * s), (x + 70 * s, ground - 270 * s)], fill=STEEL, width=max(2, int(6 * s))); d.ellipse([x + 54 * s, ground - 296 * s, x + 98 * s, ground - 252 * s], fill=CREAM)

def tower(d, x, ground, w=230, h=430):
    """A gothic college tower."""
    c = STONE
    d.rectangle([x - w / 2, ground - h, x + w / 2, ground], fill=c)
    d.rectangle([x, ground - h, x + w / 2, ground], fill=shade(c, 0.86))
    for i in range(7):
        bx = x - w / 2 + i * w / 6.5
        if i % 2 == 0: d.rectangle([bx, ground - h - 26, bx + w / 13 * 1.6, ground - h], fill=c)
    for sx in (-w / 2, w / 2):
        d.polygon([(sx + x - 16, ground - h), (sx + x, ground - h - 120), (sx + x + 16, ground - h)], fill=shade(c, 0.8))
    for row in range(3):
        for col_ in (-1, 1):
            wx, wy = x + col_ * w * 0.22, ground - h + 60 + row * 118
            d.rectangle([wx - 20, wy + 20, wx + 20, wy + 80], fill=(60, 80, 110)); d.pieslice([wx - 20, wy, wx + 20, wy + 40], 180, 360, fill=(60, 80, 110))
    d.rectangle([x - 30, ground - 76, x + 30, ground], fill=(96, 62, 40)); d.pieslice([x - 30, ground - 106, x + 30, ground - 46], 180, 360, fill=(96, 62, 40))

def temple(d, x, ground, w=400, h=330):
    """A stepped Mesoamerican temple with a central stair and a shrine on top."""
    c = (196, 112, 72); n = 5
    for i in range(n):
        a = w / 2 * (1 - i * 0.16); y0, y1 = ground - h * 0.8 * i / n, ground - h * 0.8 * (i + 1) / n
        d.rectangle([x - a, y1, x + a, y0], fill=c if i % 2 == 0 else shade(c, 0.88))
    d.polygon([(x - 46, ground), (x - 30, ground - h * 0.8), (x + 30, ground - h * 0.8), (x + 46, ground)], fill=shade(c, 0.7))
    for i in range(12): d.line([(x - 44 + i * 0.4, ground - i * h * 0.8 / 12), (x + 44 - i * 0.4, ground - i * h * 0.8 / 12)], fill=shade(c, 0.5), width=3)
    d.rectangle([x - 62, ground - h, x + 62, ground - h * 0.8], fill=shade(c, 0.8)); d.rectangle([x - 18, ground - h + 16, x + 18, ground - h * 0.8], fill=INK)
    d.rectangle([x - 74, ground - h - 16, x + 74, ground - h], fill=GOLD)

def hourglass(d, cx, cy, hw, hh, t, period=6.0):
    k = (t / period) % 1
    d.rounded_rectangle([cx - hw - 26, cy - hh - 30, cx + hw + 26, cy - hh], 12, fill=GOLD); d.rounded_rectangle([cx - hw - 26, cy + hh, cx + hw + 26, cy + hh + 30], 12, fill=GOLD)
    glass = mix(BG1, CREAM, 0.22)
    d.polygon([(cx - hw, cy - hh), (cx + hw, cy - hh), (cx + 14, cy), (cx - 14, cy)], fill=glass); d.polygon([(cx - 14, cy), (cx + 14, cy), (cx + hw, cy + hh), (cx - hw, cy + hh)], fill=glass)
    a = (1 - k) ** 0.5; ty = cy - hh * a                                             # sand left in the top
    d.polygon([(cx - (hw - 8) * a - 6, ty), (cx + (hw - 8) * a + 6, ty), (cx + 10, cy - 4), (cx - 10, cy - 4)], fill=SAND)
    b = k ** 0.5; by = cy + hh - 6                                                   # pile at the bottom
    d.polygon([(cx - (hw - 10) * b - 8, by), (cx + (hw - 10) * b + 8, by), (cx, by - hh * 0.72 * b)], fill=SAND)
    d.line([(cx, cy - 4), (cx, by)], fill=SAND, width=7)
    for sx in (-1, 1): d.line([(cx + sx * (hw + 10), cy - hh), (cx + sx * (hw + 10), cy + hh)], fill=shade(GOLD, 0.8), width=10)

# ---------- visuals (the area between y=640 and y=1330) ----------
VY0, VY1 = 640, 1330
def visual(d, kind, t, dur, img, sc=None):
    cx, ground = W / 2, VY1 - 40
    d.line([(60, ground), (W - 60, ground)], fill=LINE, width=4)
    if kind == 'standoff':
        emu(d, cx + 170, ground, 1.35, t * 0.8, face=-1)
        p = back(t / 0.5)
        text_c(d, (cx - 170, ground - 260), 'VS', font('anton', 170 * max(p, 0.01)), RED, stroke=6)
        for i in range(3):
            bullet(d, cx - 300 + i * 70, ground - 70, 1.6)
    elif kind == 'wheat':
        rng = random.Random(4)
        for i in range(34):
            x = 80 + i * 28 + rng.uniform(-6, 6); h = rng.uniform(180, 330) * ease(t / 0.8 - i * 0.01)
            sway = math.sin(t * 2.2 + i * 0.5) * 14
            d.line([(x, ground), (x + sway, ground - h)], fill=(196, 160, 80), width=5)
            for k in range(6):
                gy = ground - h + k * 16
                d.ellipse([x + sway - 8, gy - 10, x + sway + 8, gy + 10], fill=GOLD)
    elif kind in ('flock', 'run'):
        rng = random.Random(7)
        n = 26 if kind == 'flock' else 10
        speed = 210 if kind == 'flock' else 520
        for i in range(n):
            depth = rng.uniform(0.35, 1.0) if kind == 'flock' else rng.uniform(0.55, 1.0)
            yb = ground - (1 - depth) * 520
            start = rng.uniform(-W, W)
            x = (start + t * speed * depth) % (W + 400) - 200
            if kind == 'flock' and t < i * 0.08: continue
            emu(d, x, yb, 0.75 * depth, t * (1.4 if kind == 'run' else 0.9) + i * 0.37, face=1, run=kind == 'run',
                col=tuple(int(c * (0.65 + 0.35 * depth)) for c in EMU_C))
            if kind == 'run':
                for k in range(3):
                    lx = x - 130 * depth - k * 40
                    d.line([(lx, yb - (110 + k * 30) * depth), (lx - 90 * depth, yb - (110 + k * 30) * depth)], fill=(150, 130, 100), width=4)
        if kind == 'run':                                               # speedometer
            gx, gy, r = W - 190, VY0 + 150, 120
            d.arc([gx - r, gy - r, gx + r, gy + r], 150, 390, fill=CREAM, width=10)
            a = math.radians(150 + 240 * 0.8 * ease(t / 1.2))
            d.line([(gx, gy), (gx + math.cos(a) * r * 0.85, gy + math.sin(a) * r * 0.85)], fill=RED, width=10)
            d.ellipse([gx - 14, gy - 14, gx + 14, gy + 14], fill=RED)
    elif kind == 'ammo':
        cols, rows = 14, 4
        shown = int(cols * rows * ease(t / max(dur * 0.6, 0.5)))
        for i in range(shown):
            r, c = divmod(i, cols)
            bullet(d, 130 + c * 63, VY0 + 130 + r * 88, 1.3)
        gun(d, cx - 150, ground - 110, 0.75); gun(d, cx + 250, ground - 95, 0.55, col=(60, 60, 60))
    elif kind == 'jam':
        sh = math.sin(t * 60) * 10 * (t > 0.6)
        gun(d, cx - 40 + sh, VY0 + 330, 1.1)
        for i in range(12):
            bullet(d, 140 + i * 72, ground - 40, 1.0, col=GOLD if i < 11 else RED)
        if t > 0.7:
            p = back((t - 0.7) / 0.35)
            img.alpha_composite(stamp('JAMMED!', p), (int((cx - 330 + LAYOUT['vox']) * S), int((VY0 + 360 + LAYOUT['voy']) * S)))
    elif kind == 'ratio':
        k = int(10 * ease(t / 1.4))
        for i in range(k):
            bullet(d, 110 + i * 62, VY0 + 150, 1.5)
        text_c(d, (cx, VY0 + 320), '=', font('anton', 200), CREAM)
        if t > 1.2: emu(d, cx, ground, 0.7, t, face=-1)
    elif kind == 'tank':
        buildings(d, 80, ground, n=2, seed=3, scale=0.8); buildings(d, W - 330, ground, n=2, seed=6, scale=0.8)
        tank(d, cx, ground, 380, 500, level=ease(t / max(dur * 0.7, 0.5)))
    elif kind == 'crack':
        k = ease((t - 0.8) / 1.6); sh = math.sin(t * 55) * 9 * (t > 0.8)
        tank(d, cx, ground, 400, 520, level=0.95, shake=sh, crack=k)
        for i in range(int(8 * k)):                                  # leaks
            dx = cx - 170 + i * 48; dy = (t * 260 + i * 70) % 300
            d.ellipse([dx - 8, ground - 300 + dy - 12, dx + 8, ground - 300 + dy + 12], fill=LIQ)
    elif kind == 'wave':
        p = ease(t / max(dur * 0.9, 1)); front = -60 + p * 760
        buildings(d, 470, ground, n=5, seed=2, tilt=max(0, p - 0.45) * 1.2)
        wave(d, front, ground, 420, t)
    elif kind == 'speed':
        gx, gy, r = cx, VY0 + 330, 250
        d.arc([gx - r, gy - r, gx + r, gy + r], 150, 390, fill=CREAM, width=18)
        for i in range(9):
            a = math.radians(150 + 240 * i / 8)
            d.line([(gx + math.cos(a) * r * 0.82, gy + math.sin(a) * r * 0.82), (gx + math.cos(a) * r * 0.95, gy + math.sin(a) * r * 0.95)], fill=CREAM, width=8)
        a = math.radians(150 + 240 * 0.7 * back(t / 1.3))
        d.line([(gx, gy), (gx + math.cos(a) * r * 0.78, gy + math.sin(a) * r * 0.78)], fill=RED, width=18)
        d.ellipse([gx - 26, gy - 26, gx + 26, gy + 26], fill=RED)
        wave(d, (t * 700) % (W + 500) - 150, ground, 150, t, left=-400)
    elif kind == 'toll':
        n = int(21 * ease(t / max(dur * 0.6, 0.5)))
        for i in range(n):
            r, c = divmod(i, 7)
            person(d, 180 + c * 120, VY0 + 190 + r * 170, 1.5, CREAM)
        d.rectangle([60, ground - 46, W - 60, ground], fill=LIQ)
    elif kind == 'law':
        buildings(d, 90, ground, n=3, seed=8, scale=0.7)
        d.rectangle([60, ground - 30 * (1 - ease(t / max(dur, 1))), W - 60, ground], fill=LIQ)   # the mess draining away
        d.rounded_rectangle([cx + 40, VY0 + 110, cx + 420, ground - 60], 14, fill=CREAM)
        for i in range(7):
            d.line([(cx + 80, VY0 + 180 + i * 56), (cx + (380 if i % 3 else 300), VY0 + 180 + i * 56)], fill=shade(STEEL, 0.8), width=9)
        if t > 0.9:
            img.alpha_composite(stamp('SAFETY', back((t - 0.9) / 0.35)), (int((cx - 250 + LAYOUT['vox']) * S), int((VY0 + 330 + LAYOUT['voy']) * S)))
    elif kind == 'follow_wave':
        p = 1 + 0.06 * math.sin(t * 6)
        bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 250 - bh / 2, cx + bw / 2, VY0 + 250 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 250), 'FOLLOW', font('anton', 96 * p), CREAM)
        wave(d, (t * 420) % (W + 500) - 150, ground, 170, t, left=-400)
    elif kind == 'dance1':
        sparkles(d, t)
        d.ellipse([cx - 230, ground - 26, cx + 230, ground + 26], fill=mix(BG1, GOLD, 0.35))        # pool of light
        dancer(d, cx, ground, 1.65, t * 1.3, POP[0], style=1)
        notes(d, t, 120, W - 120, VY0 + 40, ground - 250, n=6)
    elif kind == 'crowd':
        sparkles(d, t, 14)
        n = (sc or {}).get('n', 30); rows = (sc or {}).get('rows', 3)
        rng = random.Random(9); per = math.ceil(n / rows)
        shown = int(n * ease(t / max(dur * 0.7, 0.6))) + 1
        k = 0
        for r in range(rows):                                       # back rows first: smaller and dimmer
            depth = (r + 1) / rows; sc_ = (2.3 / (per ** 0.62)) * (0.62 + 0.38 * depth)
            yb = ground - (rows - 1 - r) * (470 / rows)
            for c in range(per):
                if k >= min(n, shown): break
                x = 70 + (W - 140) * (c + 0.5 + (0.5 if r % 2 else 0) * 0.6) / per
                col = mix(BG1, POP[rng.randrange(len(POP))], 0.45 + 0.55 * depth)
                dancer(d, x, yb, sc_, t * rng.uniform(1.0, 1.6) + rng.random(), col, style=k)
                k += 1
    elif kind == 'fever':
        sparkles(d, t, 12)
        x0, top, bot = cx - 250, VY0 + 110, ground - 110
        d.rounded_rectangle([x0 - 34, top, x0 + 34, bot], 34, fill=CREAM)                 # thermometer
        d.ellipse([x0 - 78, bot - 60, x0 + 78, bot + 96], fill=CREAM)
        lvl = ease(t / 1.4); fy = bot - (bot - top - 60) * lvl
        d.rounded_rectangle([x0 - 15, fy, x0 + 15, bot + 10], 15, fill=RED)
        d.ellipse([x0 - 54, bot - 36, x0 + 54, bot + 72], fill=RED)
        for i in range(6):
            yy = top + 70 + i * (bot - top - 130) / 5
            d.line([(x0 + 44, yy), (x0 + 84, yy)], fill=CREAM, width=7)
        dancer(d, cx + 190, ground, 1.5, t * 1.5, POP[2], style=0)
        for i in range(3):                                                             # heat waves
            hx = cx + 60 + i * 130; ph = t * 3 + i
            pts = [(hx + math.sin(ph + j * 0.9) * 16, ground - 520 - j * 22) for j in range(6)]
            d.line(pts, fill=mix(BG1, RED, 0.8), width=8, joint='curve')
    elif kind == 'band':
        d.polygon([(90, ground), (150, ground - 120), (W - 150, ground - 120), (W - 90, ground)], fill=mix(BG1, GOLD, 0.3))   # stage
        d.rectangle([150, ground - 132, W - 150, ground - 116], fill=GOLD)
        sy = ground - 126
        dancer(d, cx - 40, sy, 1.15, t * 1.5, POP[0], style=1); dancer(d, cx + 150, sy, 1.15, t * 1.5 + 0.4, POP[1], style=2)
        dx = 255; beat = abs(math.sin(t * 7)) * 16                                      # drummer
        dancer(d, dx - 70, sy, 0.95, t * 0.6, POP[3], style=0)
        d.ellipse([dx - 10, sy - 150 + beat * 0.2, dx + 130, sy - 40], fill=CREAM); d.ellipse([dx + 4, sy - 138 + beat * 0.2, dx + 116, sy - 52], fill=RED)
        hx = W - 300                                                                    # horn player
        dancer(d, hx + 90, sy, 0.95, t * 0.6 + 0.3, POP[4], style=0)
        d.polygon([(hx + 60, sy - 196), (hx - 70, sy - 250 - beat), (hx - 70, sy - 150 + beat)], fill=GOLD)
        notes(d, t, 120, W - 120, VY0 + 20, sy - 300, n=9)
    elif kind == 'calendar':
        sparkles(d, t, 12)
        for i, mth in enumerate((sc or {}).get('months', ['JUL', 'AUG', 'SEP'])):
            p = back((t - 0.25 - i * 0.55) / 0.4)
            if p <= 0: continue
            x = cx - 330 + i * 330; yy = VY0 + 270 + (1 - p) * 80; wd, hg = 135, 160
            d.rounded_rectangle([x - wd, yy - hg, x + wd, yy + hg], 26, fill=CREAM)
            d.rounded_rectangle([x - wd, yy - hg, x + wd, yy - hg + 96], 26, fill=POP[i % len(POP)])
            d.rectangle([x - wd, yy - hg + 60, x + wd, yy - hg + 96], fill=POP[i % len(POP)])
            text_c(d, (x, yy - hg + 50), str((sc or {}).get('year', '1518')), font('inter', 40), INK)
            text_c(d, (x, yy + 50), mth, font('anton', 130), INK)
        if (sc or {}).get('snow'): snowfall(d, t, 50)
        else:
            for i in range(5):
                dancer(d, 150 + i * 195, ground, 0.62, t * 1.4 + i * 0.3, POP[i % len(POP)], style=i)
    elif kind == 'mystery':
        sparkles(d, t)
        p = 1 + 0.07 * math.sin(t * 4)
        text_c(d, (cx, VY0 + 300), '?', font('anton', 560 * p), GOLD, stroke=12)
        for i in range(6):
            a = t * 0.9 + i * math.pi / 3
            x, yy = cx + math.cos(a) * 400, VY0 + 330 + math.sin(a) * 150
            depth = 0.5 + 0.5 * math.sin(a)
            dancer(d, x, yy + 130, 0.45 + 0.2 * depth, t * 1.4 + i, mix(BG1, POP[i % len(POP)], 0.5 + 0.5 * depth), style=i)
    elif kind == 'follow_dance':
        sparkles(d, t)
        p = 1 + 0.06 * math.sin(t * 6)
        bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 170 - bh / 2, cx + bw / 2, VY0 + 170 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 170), 'FOLLOW', font('anton', 96 * p), CREAM)
        for i in range(5):
            dancer(d, 150 + i * 195, ground, 0.95, t * 1.5 + i * 0.25, POP[i % len(POP)], style=i)
    elif kind == 'volcano_quiet':
        sea(d, ground); ty = volcano(d, cx, ground, crater=(sc or {}).get('crater', False), h=(sc or {}).get('h', 500), top=(sc or {}).get('top', 0.13))
        for i in range(4):                                         # a lazy wisp of smoke
            a = (t * 0.12 + i / 4) % 1
            d.ellipse([cx - 20 + a * 90 - 16 - a * 30, ty - 30 - a * 200 - 16 - a * 30, cx - 20 + a * 90 + 16 + a * 30, ty - 30 - a * 200 + 16 + a * 30], fill=mix(ASH, BG1, a))
        for x_, s_ in ((110, 0.8), (190, 1.0), (930, 0.9)): tree(d, x_, ground, s_, bare=False)
        sparkles(d, t, 10)
    elif kind == 'erupt':
        pw = (sc or {}).get('power', 1.0); sh = math.sin(t * 50) * 7 * pw
        d.ellipse([cx - 430, ground - 70, cx + 430, ground + 60], fill=mix(BG1, FIRE, 0.35))
        sea(d, ground); ty = volcano(d, cx, ground, shake=sh)
        if (sc or {}).get('columns'):
            for dx in (-70, 0, 70): plume(d, cx + dx + sh, ty + 20, t + dx, 0.72, n=14)
        else: plume(d, cx + sh, ty + 20, t, pw)
        lava(d, cx + sh, ty, ground, 760, t); rocks(d, cx, ty, t, power=pw)
    elif kind == 'collapse':
        k = ease((t - 1.0) / 2.2); h = 500 - 170 * k
        sea(d, ground)
        for i in range(12):                                         # dotted outline of the old peak
            u = i / 11; x0 = cx - 380 + 280 * u; x1 = cx + 380 - 280 * u
            for xx in (x0, x1): d.ellipse([xx - 5, ground - 500 * u - 5, xx + 5, ground - 500 * u + 5], fill=CREAM)
        ty = volcano(d, cx, ground, h=h, top=0.13 + 0.12 * k, crater=k > 0.5)
        if k < 1: plume(d, cx, ty + 10, t, 0.5 + 0.4 * (1 - k), n=18)
        d.line([(W - 110, ground), (W - 110, ground - 500)], fill=CREAM, width=5)
        for yy in (0, 500): d.line([(W - 130, ground - yy), (W - 90, ground - yy)], fill=CREAM, width=5)
        d.line([(W - 140, ground - h), (W - 80, ground - h)], fill=GOLD, width=9)
    elif kind == 'boom':
        vx = 240; sea(d, ground); ty = volcano(d, vx, ground, w=400, h=270)
        plume(d, vx, ty + 10, t, 0.5, n=16)
        for k in range(6):
            r = (t * 300 + k * 170) % 1020
            if r < 60: continue
            d.arc([vx - r, ty - r, vx + r, ty + r], -38, 22, fill=mix(GOLD, BG1, min(1, r / 1020)), width=10)
        for i in range(3):
            person(d, 860 + i * 70, ground - 34, 1.5, POP[2 % len(POP)])
        if (t * 300) % 1020 > 560 or t > 3: text_c(d, (930, ground - 190 + math.sin(t * 9) * 6), '!?' if (sc or {}).get('ask') else '!', font('anton', 120), GOLD, stroke=6)
    elif kind == 'ash_dark':
        d.ellipse([cx - 80, VY0 + 80, cx + 80, VY0 + 240], fill=mix(BG1, FIRE, 0.35))                 # the sun, barely visible
        house(d, 150, ground, 200, 150, shade(STEEL, 0.7), shade(STEEL, 0.5), lit=False); house(d, 660, ground, 260, 180, shade(STEEL, 0.7), shade(STEEL, 0.5), lit=False)
        tree(d, 500, ground, 1.0)
        lvl = 30 + 120 * ease(t / max(dur, 1))
        pts = [(0, ground + 40)] + [(x, ground - lvl - math.sin(x * 0.012) * 26) for x in range(0, W + 1, 60)] + [(W, ground + 40)]
        d.polygon(pts, fill=ASH)
        snowfall(d, t, 110, col=shade(ASH, 1.5), dots=True)
    elif kind == 'people':
        n = (sc or {}).get('n', 40); cols = 10; shown = int(n * ease(t / max(dur * 0.6, 0.6)))
        for i in range(shown):
            r, c = divmod(i, cols)
            person(d, 100 + c * 98, VY0 + 150 + r * 150, 1.25, mix(CREAM, BG1, 0.55) if (i * 7) % 10 < (sc or {}).get('fade', 0) else CREAM)
    elif kind == 'ash_globe':
        R = 250; gy = VY0 + 345
        d.ellipse([cx - R, gy - R, cx + R, gy + R], fill=LIQ)
        for bx, by, bw, bh in ((-120, -90, 110, 80), (-60, 40, 70, 100), (60, -120, 130, 70), (110, 10, 90, 120), (20, 130, 80, 50), (-170, 40, 50, 60)):
            d.ellipse([cx + bx - bw / 2, gy + by - bh / 2, cx + bx + bw / 2, gy + by + bh / 2], fill=LEAF)
        tx_, ty_ = cx + 150, gy + 60
        d.ellipse([tx_ - 12, ty_ - 12, tx_ + 12, ty_ + 12], fill=FIRE)
        k = ease(t / max(dur * 0.85, 1)); rng = random.Random(3)
        for i in range(150):
            u = i / 150
            if u > k: break
            a = math.radians(-20) + u * 2 * math.pi + t * 0.15
            rr = R + 36 + rng.uniform(-30, 30); r = rng.uniform(22, 40)
            x, y = cx + math.cos(a) * rr * 1.08, gy + math.sin(a) * rr * 0.5 + math.cos(a) * 40
            if math.sin(a) < -0.15 and abs(math.cos(a)) < 0.92: continue                     # hidden behind the globe
            d.ellipse([x - r, y - r, x + r, y + r], fill=mix(ASH, CREAM, rng.uniform(0.1, 0.35)))
        sparkles(d, t, 8)
    elif kind == 'dim_sun':
        k = ease(t / max(dur * 0.8, 1)); sy_ = VY0 + 280; col = mix(HOT, (190, 96, 60), k)
        for i in range(12):
            a = i * math.pi / 6 + t * 0.2; r0, r1 = 190, 190 + 110 * (1 - 0.75 * k)
            d.line([(cx + math.cos(a) * r0, sy_ + math.sin(a) * r0), (cx + math.cos(a) * r1, sy_ + math.sin(a) * r1)], fill=col, width=14)
        d.ellipse([cx - 160, sy_ - 160, cx + 160, sy_ + 160], fill=col)
        for i in range(int(7 * k) + 1):                                                       # bands of haze
            yy = sy_ - 130 + i * 44; off = math.sin(t * 0.6 + i) * 60
            d.rounded_rectangle([cx - 330 + off, yy, cx + 330 + off, yy + 16], 8, fill=mix(ASH, BG1, 0.25))
        d.rectangle([0, ground - 30, W, ground + 40], fill=mix(LEAF, BG0, 0.35 + 0.4 * k))
    elif kind == 'cold':
        x0, top, bot = cx - 220, VY0 + 90, ground - 110
        d.rounded_rectangle([x0 - 34, top, x0 + 34, bot], 34, fill=CREAM); d.ellipse([x0 - 78, bot - 60, x0 + 78, bot + 96], fill=CREAM)
        lvl = 0.78 - 0.4 * ease(t / 2.4); col = mix(FIRE, (70, 160, 255), ease(t / 2.4)); fy = bot - (bot - top - 60) * lvl
        d.rounded_rectangle([x0 - 15, fy, x0 + 15, bot + 10], 15, fill=col); d.ellipse([x0 - 54, bot - 36, x0 + 54, bot + 72], fill=col)
        for i in range(6):
            yy = top + 70 + i * (bot - top - 130) / 5; d.line([(x0 + 44, yy), (x0 + 84, yy)], fill=CREAM, width=7)
        for i in range(5):
            flake(d, cx + 90 + (i % 3) * 150, VY0 + 170 + (i // 3) * 220 + math.sin(t * 2 + i) * 14, 46 + 8 * math.sin(t * 3 + i), SNOW)
    elif kind == 'snow_farm':
        k = ease(t / max(dur * 0.8, 1))
        d.rectangle([0, ground, W, ground + 40], fill=mix(LEAF, SNOW, k))
        house(d, 170, ground, 250, 190, (150, 84, 70), (92, 60, 60), snow=k); house(d, 600, ground, 190, 130, (170, 60, 52), (96, 50, 50), lit=False, snow=k)
        tree(d, 900, ground, 1.25); tree(d, 520, ground, 0.8)
        pts = [(0, ground + 40)] + [(x, ground - 34 * k - math.sin(x * 0.02) * 10 * k) for x in range(0, W + 1, 60)] + [(W, ground + 40)]
        d.polygon(pts, fill=SNOW)
        snowfall(d, t, 70)
        cal_card(d, 930, VY0 + 150, (sc or {}).get('month', 'JUN'), 1816, POP[2 % len(POP)], 0.62)
    elif kind == 'crops':
        k = ease(t / max(dur * 0.7, 1)); rng = random.Random(4)
        for i in range(34):
            x = 80 + i * 28 + rng.uniform(-6, 6); h = rng.uniform(190, 340); bend = k * rng.uniform(60, 170) * (1 if i % 2 else -1)
            col = mix(HOT, (120, 120, 128), k)
            pts = [(x, ground), (x + bend * 0.25, ground - h * 0.6), (x + bend, ground - h * (1 - 0.45 * k))]
            d.line(pts, fill=mix((196, 160, 80), (110, 110, 118), k), width=5, joint='curve')
            for j in range(5):
                gx, gy = pts[2][0] - bend * 0.12 * j, pts[2][1] + j * 15
                d.ellipse([gx - 8, gy - 10, gx + 8, gy + 10], fill=col)
        snowfall(d, t, int(40 * k) + 4)
    elif kind == 'price':
        bx0, bw, n = 200, 96, 6; hs = [80, 120, 170, 250, 350, 470]
        d.line([(bx0 - 40, VY0 + 80), (bx0 - 40, ground)], fill=CREAM, width=6)
        tops = []
        for i in range(n):
            p = ease((t - i * 0.35) / 0.5); h = hs[i] * p; x = bx0 + i * (bw + 28)
            if p <= 0: continue
            d.rounded_rectangle([x, ground - h, x + bw, ground], 10, fill=mix(HOT, FIRE, i / (n - 1)))
            tops.append((x + bw / 2, ground - h - 36))
        if len(tops) > 1:
            d.line(tops, fill=CREAM, width=9, joint='curve')
            (x1, y1), (x0_, y0_) = tops[-1], tops[-2]; a = math.atan2(y1 - y0_, x1 - x0_)
            d.polygon([(x1 + math.cos(a) * 40, y1 + math.sin(a) * 40), (x1 + math.cos(a + 2.4) * 30, y1 + math.sin(a + 2.4) * 30), (x1 + math.cos(a - 2.4) * 30, y1 + math.sin(a - 2.4) * 30)], fill=CREAM)
        text_c(d, (bx0 + 150, VY0 + 110), (sc or {}).get('label', 'PRICE'), font('anton', 90), CREAM, stroke=5)
    elif kind == 'riot':
        for i in range(7):
            x = 130 + i * 138; b = abs(math.sin(t * 5 + i * 1.3)) * 22; s_ = 1.9
            person(d, x, ground - 44 - b, s_, POP[i % len(POP)])
            if i % 2 == 0:
                d.line([(x + 30, ground - 80 - b), (x + 40, ground - 330 - b)], fill=(150, 104, 62), width=9)
                d.rounded_rectangle([x - 40, ground - 420 - b, x + 120, ground - 320 - b], 10, fill=CREAM)
                text_c(d, (x + 40, ground - 370 - b), 'BREAD', font('anton', 50), INK)
    elif kind == 'migrate':
        d.ellipse([W - 300, VY0 + 110, W - 140, VY0 + 270], fill=mix(HOT, (190, 96, 60), 0.7))
        for mx, mh in ((180, 190), (460, 250), (760, 170)): d.polygon([(mx - 200, ground), (mx, ground - mh), (mx + 200, ground)], fill=shade(STEEL, 0.55))
        for i in range(3):
            x = (t * 120 + i * 420) % (W + 400) - 200
            wagon(d, x, ground, 1.15, t)
    elif kind == 'rain_villa':
        for mx, mh in ((160, 300), (520, 380), (900, 280)): d.polygon([(mx - 260, ground - 30), (mx, ground - 30 - mh), (mx + 260, ground - 30)], fill=shade(STEEL, 0.5))
        d.rectangle([0, ground - 30, W, ground + 40], fill=LIQ)
        house(d, 330, ground - 30, 420, 250, (190, 170, 150), (96, 64, 64))
        for i in range(4): d.rectangle([356 + i * 96, ground - 150, 396 + i * 96, ground - 100], fill=HOT)
        tree(d, 220, ground - 30, 1.2); tree(d, 850, ground - 30, 1.0)
        fl = (t % 3.2) < 0.16 or 1.1 < (t % 3.2) < 1.2
        if fl: bolt(d, 840 + (int(t / 3.2) % 2) * -640, VY0 + 20, 1.1, SNOW)
        rain(d, t, 90)
    elif kind == 'writers':
        d.rounded_rectangle([cx - 150, VY0 + 60, cx + 150, VY0 + 300], 16, fill=mix(LIQ, BG0, 0.4))          # window
        d.line([(cx, VY0 + 60), (cx, VY0 + 300)], fill=shade(STEEL, 0.6), width=10); d.line([(cx - 150, VY0 + 180), (cx + 150, VY0 + 180)], fill=shade(STEEL, 0.6), width=10)
        rain(d, t, 16, top=VY0 + 70, bottom=VY0 + 290)
        for i in range(4):
            x = 190 + i * 235; b = math.sin(t * 2 + i) * 5
            person(d, x, ground - 190 + b, 2.3, POP[i % len(POP)])
        d.rounded_rectangle([90, ground - 150, W - 90, ground - 110], 16, fill=(132, 92, 58)); d.rectangle([150, ground - 110, 190, ground], fill=(100, 70, 46)); d.rectangle([W - 190, ground - 110, W - 150, ground], fill=(100, 70, 46))
        d.rectangle([cx - 12, ground - 230, cx + 12, ground - 150], fill=CREAM)                               # candle
        fk = 1 + 0.18 * math.sin(t * 17) * math.sin(t * 5)
        d.ellipse([cx - 20 * fk, ground - 300 * (0.94 + 0.06 * fk), cx + 20 * fk, ground - 226], fill=HOT)
        if (sc or {}).get('ghost'):
            gx, gy = 190 + 235, VY0 + 120 + math.sin(t * 2.5) * 12
            d.rounded_rectangle([gx - 110, gy - 100, gx + 110, gy + 90], 40, fill=CREAM)
            d.polygon([(gx - 30, gy + 80), (gx + 10, gy + 80), (gx - 20, gy + 130)], fill=CREAM)
            ghost(d, gx, gy, 1.35, mix(BG1, LIQ, 0.5))
    elif kind == 'lab':
        d.rounded_rectangle([cx - 330, ground - 150, cx + 330, ground - 110], 12, fill=shade(STEEL, 0.8))       # the slab
        for lx in (-280, 280): d.rectangle([cx + lx - 18, ground - 110, cx + lx + 18, ground], fill=shade(STEEL, 0.6))
        d.rounded_rectangle([cx - 290, ground - 215, cx + 250, ground - 150], 32, fill=CREAM)                  # a covered figure
        d.ellipse([cx + 200, ground - 240, cx + 290, ground - 150], fill=CREAM)
        on = (t % 1.3) < 0.5 and t > 0.8
        if on:
            bolt(d, cx + 20, VY0 + 10, 1.25)
            for i in range(10):
                a = i * 0.63 + t * 9; r = 70 + (i * 37 % 60)
                d.ellipse([cx + math.cos(a) * r * 2.2 - 8, ground - 210 + math.sin(a) * r * 0.7 - 8, cx + math.cos(a) * r * 2.2 + 8, ground - 210 + math.sin(a) * r * 0.7 + 8], fill=HOT)
        sparkles(d, t, 8)
    elif kind == 'book':
        p = back(t / 0.5); bob = math.sin(t * 1.6) * 10; bw, bh = 330 * p, 460 * p; y = VY0 + 345 + bob
        if p > 0.05:
            d.rounded_rectangle([cx - bw / 2 + 26, y - bh / 2 + 22, cx + bw / 2 + 26, y + bh / 2 + 22], 14, fill=CREAM)
            cov = tuple((sc or {}).get('cover', [120, 28, 44]))
            d.rounded_rectangle([cx - bw / 2, y - bh / 2, cx + bw / 2, y + bh / 2], 14, fill=cov)
            if bw > 120: d.rounded_rectangle([cx - bw / 2 + 22, y - bh / 2 + 22, cx + bw / 2 - 22, y + bh / 2 - 22], 8, outline=HOT, width=5)
            title = (sc or {}).get('title', 'BOOK'); words_ = title.split()
            fs = min(86, 500 / max(len(w_) for w_ in words_)) * p
            for i, w_ in enumerate(words_):
                text_c(d, (cx, y - 70 + (i - (len(words_) - 1) / 2) * fs * 1.15), w_, font('anton', fs), HOT)
            text_c(d, (cx, y + bh / 2 - 80), str((sc or {}).get('year', '')), font('inter', 44 * p), CREAM)
        if t > 0.7 and (t % 2.4) < 0.3: bolt(d, cx + 330, VY0 + 60, 0.9)
        sparkles(d, t, 14)
    elif kind == 'bats':
        d.ellipse([W - 400, VY0 + 60, W - 140, VY0 + 320], fill=CREAM); d.ellipse([W - 330, VY0 + 110, W - 280, VY0 + 160], fill=shade(CREAM, 0.88)); d.ellipse([W - 250, VY0 + 210, W - 190, VY0 + 270], fill=shade(CREAM, 0.88))
        d.polygon([(0, ground + 40), (0, ground - 120), (80, ground - 120), (80, ground - 300), (120, ground - 380), (160, ground - 300), (160, ground - 160), (260, ground - 160),
                   (260, ground - 240), (300, ground - 300), (340, ground - 240), (340, ground - 60), (W, ground - 20), (W, ground + 40)], fill=shade(STEEL, 0.42))
        d.rectangle([104, ground - 270, 134, ground - 220], fill=HOT)
        for i in range(8):
            x = (t * (110 + i * 22) + i * 190) % (W + 300) - 150; y = VY0 + 150 + (i * 97 % 330) + math.sin(t * 2 + i) * 30
            bat(d, x, y, 0.6 + (i % 3) * 0.3, t * 13 + i, col=mix(INK, BG1, 0.25 * (i % 3)))
    elif kind == 'sunset':
        x0, y0, x1, y1 = cx - 330, VY0 + 60, cx + 330, VY0 + 470
        d.line([(cx - 200, y1), (cx - 300, ground)], fill=(132, 92, 58), width=16); d.line([(cx + 200, y1), (cx + 300, ground)], fill=(132, 92, 58), width=16); d.line([(cx, y1), (cx, ground - 20)], fill=(110, 78, 50), width=14)
        d.rectangle([x0 - 22, y0 - 22, x1 + 22, y1 + 22], fill=(150, 104, 62))
        d.rectangle([x0, y0, x1, y1], fill=CREAM)
        bands = [(96, 30, 60), (150, 36, 56), (204, 52, 48), (234, 84, 40), (250, 124, 40), (255, 164, 56), (255, 200, 84)]
        nb = len(bands); hz = y0 + (y1 - y0) * 0.72; shown = ease(t / max(dur * 0.6, 1)) * nb
        for i, c in enumerate(bands):
            if i > shown: break
            d.rectangle([x0, y0 + (hz - y0) * i / nb, x1, y0 + (hz - y0) * (i + 1) / nb + 1], fill=c)
        if shown >= nb - 0.5:
            d.pieslice([cx - 90, hz - 90, cx + 90, hz + 90], 180, 360, fill=HOT)
            d.rectangle([x0, hz, x1, y1], fill=(60, 34, 60))
            for i in range(4): d.line([(cx - 70 + i * 10, hz + 22 + i * 22), (cx + 70 - i * 10, hz + 22 + i * 22)], fill=(250, 124, 40), width=6)
    elif kind == 'bike':
        for i in range(4):
            x = (i * 280 - t * 230) % 1120 - 120; tree(d, x, ground, 1.0, bare=False)      # stay inside the visual area
        x = cx - 40 + math.sin(t * 0.5) * 130                      # the rider stays in view; the trees roll past
        draisine(d, x, ground, t, POP[0])
        for i in range(3): d.line([(x - 240 - i * 50, ground - 60 - i * 40), (x - 320 - i * 50, ground - 60 - i * 40)], fill=mix(CREAM, BG1, 0.5), width=5)
    elif kind == 'timeline':
        nodes = (sc or {}).get('nodes', ['A', 'B', 'C']); n = len(nodes); y = VY0 + 340
        xs = [130 + (W - 260) * i / max(n - 1, 1) for i in range(n)]
        k = ease(t / max(dur * 0.55, 1)); d.line([(xs[0], y), (xs[0] + (xs[-1] - xs[0]) * k, y)], fill=CREAM, width=10)
        for i, (x, lab) in enumerate(zip(xs, nodes)):
            p = back((t - 0.3 - i * (max(dur * 0.55, 1) / n)) / 0.4)
            if p <= 0: continue
            r = 34 * p; d.ellipse([x - r, y - r, x + r, y + r], fill=POP[i % len(POP)]); d.ellipse([x - r * 0.4, y - r * 0.4, x + r * 0.4, y + r * 0.4], fill=INK)
            up = -1 if i % 2 == 0 else 1
            d.line([(x, y + up * 40), (x, y + up * 110)], fill=CREAM, width=5)
            for j, part in enumerate(lab.split('|')):
                text_c(d, (min(max(x, 190), W - 190), y + up * 160 + (j - (len(lab.split('|')) - 1) / 2) * 70 * 1 + (0 if up < 0 else 0)), part, font('anton', 62 * min(p, 1)), POP[i % len(POP)] if j == 0 else CREAM, stroke=5)
    elif kind == 'question':
        sparkles(d, t)
        p = 1 + 0.07 * math.sin(t * 4)
        text_c(d, (cx, VY0 + 330), '?', font('anton', 600 * p), GOLD, stroke=12)
        for i in range(6):
            a = t * 0.8 + i * math.pi / 3
            text_c(d, (cx + math.cos(a) * 390, VY0 + 340 + math.sin(a) * 200), '?', font('anton', 90 + 40 * math.sin(a)), mix(BG1, POP[i % len(POP)], 0.6 + 0.4 * math.sin(a) ** 2))
    elif kind == 'subscribe':
        sea(d, ground); ty = volcano(d, cx, ground, w=560, h=250); plume(d, cx, ty + 10, t, 0.36, n=14); rocks(d, cx, ty, t, n=8, power=0.4)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 640 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 120 - bh / 2, cx + bw / 2, VY0 + 120 + bh / 2], int(bh / 2), fill=(230, 40, 40))
        text_c(d, (cx, VY0 + 120), 'SUBSCRIBE', font('anton', 96 * p), SNOW)
    elif kind == 'hourglass':
        sparkles(d, t)
        hourglass(d, cx, VY0 + 350, 170, 250, t)
    elif kind == 'timebar':
        y = VY0 + 330; x0, x1 = 100, 980; px = (x1 - x0) / 4529.0; xc = x0 + 2491 * px
        k = ease(t / 1.6); d.line([(x0, y), (x0 + (x1 - x0) * k, y)], fill=CREAM, width=10)
        for i, (x, lab) in enumerate(((x0, '2560 BC'), (xc, '69 BC'), (x1, '1969'))):
            p = back((t - 0.2 - i * 0.6) / 0.4)
            if p <= 0.02: continue
            r = 26 * p; d.ellipse([x - r, y - r, x + r, y + r], fill=POP[i % len(POP)])
            text_c(d, (x, y + 70), lab, font('anton', 50 * min(p, 1)), CREAM, stroke=4)
            if i == 0: pyramid(d, x + 40, y - 50, 230 * p, 170 * p)
            elif i == 1: queen(d, x, y - 46, 0.62 * p)
            else:
                d.ellipse([x - 110 * p, y - 250 * p, x + 10 * p, y - 130 * p], fill=shade(STONE, 1.05)); d.ellipse([x - 80 * p, y - 220 * p, x - 56 * p, y - 196 * p], fill=shade(STONE, 0.85))
                lander(d, x - 40, y - 46, 0.42 * p)
        for i, (xa, xb, lab) in enumerate(((x0, xc, '2,500 YEARS'), (xc, x1, '2,000 YEARS'))):
            p = ease((t - 2.2 - i * 0.9) / 0.6)
            if p <= 0: continue
            yy = y + 170 + i * 0; col = POP[i % len(POP)] if i == 0 else POP[2 % len(POP)]
            d.rounded_rectangle([xa + 8, yy, xa + 8 + (xb - xa - 16) * p, yy + 26], 13, fill=col)
            if p > 0.6: text_c(d, ((xa + xb) / 2, yy + 80), lab, font('anton', 62), col, stroke=5)
    elif kind == 'pyramid':
        d.ellipse([W - 330, VY0 + 60, W - 130, VY0 + 260], fill=HOT)
        d.polygon([(0, ground + 40), (0, ground - 40), (260, ground - 90), (560, ground - 30), (820, ground - 80), (W, ground - 30), (W, ground + 40)], fill=shade(SAND, 0.62))
        k = ease(t / max(dur * 0.75, 1.2))
        pyramid(d, cx + 250, ground - 40, 330, 250, k=1.0, col=shade(SAND, 0.82)); pyramid(d, cx - 60, ground, 640, 480, k=k, n=12)
        for i in range(3): person(d, 120 + i * 60, ground - 26, 0.95, shade(SAND, 0.45))
    elif kind == 'moonwalk':
        sparkles(d, t, 30)
        ex, ey = 210, VY0 + 170
        d.ellipse([ex - 90, ey - 90, ex + 90, ey + 90], fill=(44, 110, 190))
        for bx, by, bw, bh in ((-30, -30, 60, 50), (30, 20, 50, 60), (-40, 40, 40, 30)): d.ellipse([ex + bx - bw / 2, ey + by - bh / 2, ex + bx + bw / 2, ey + by + bh / 2], fill=LEAF)
        d.ellipse([-260, ground - 110, W + 260, ground + 700], fill=shade(STONE, 0.78))
        for cx_, cy_, r in ((200, ground + 10, 70), (640, ground - 50, 44), (900, ground + 30, 90), (420, ground + 60, 36)):
            d.ellipse([cx_ - r, cy_ - r * 0.36, cx_ + r, cy_ + r * 0.36], fill=shade(STONE, 0.6))
        lander(d, 720, ground - 84, 1.25)
        d.line([(430, ground - 70), (430, ground - 330)], fill=CREAM, width=8); d.polygon([(434, ground - 330), (560, ground - 310 + math.sin(t * 2) * 5), (434, ground - 250)], fill=RED)
        hop = abs(math.sin(t * 1.5)) * 90
        person(d, 250 + math.sin(t * 0.75) * 60, ground - 110 - hop, 2.2, SNOW)
    elif kind == 'mammoth':
        isl = (sc or {}).get('island', False)
        if isl:
            d.rectangle([0, ground - 10, W, ground + 40], fill=(40, 96, 150))
            d.polygon([(90, ground + 40), (150, ground - 40), (400, ground - 70), (760, ground - 60), (950, ground - 30), (1000, ground + 40)], fill=SNOW)
            mammoth(d, 360 + math.sin(t * 0.5) * 30, ground - 56, 0.8, t); mammoth(d, 700 + math.sin(t * 0.4 + 1) * 24, ground - 50, 0.55, t + 1)
        else:
            d.polygon([(0, ground + 40), (0, ground - 20), (300, ground - 44), (700, ground - 14), (W, ground - 40), (W, ground + 40)], fill=SNOW)
            mammoth(d, cx - 110 + math.sin(t * 0.5) * 40, ground - 22, 1.3, t)
            p = back((t - 1.0) / 0.4)
            if p > 0.02:                                                             # meanwhile, far away
                x0, y0, wd, hg = W - 380, VY0 + 20, 340 * p, 250 * p
                d.rounded_rectangle([x0, y0, x0 + wd, y0 + hg], 22, fill=shade(SAND, 0.5))
                if p > 0.9:
                    pyramid(d, x0 + wd / 2, y0 + hg - 56, 210, 140, k=ease((t - 1.4) / 2.5))
                    text_c(d, (x0 + wd / 2, y0 + hg - 26), 'MEANWHILE IN EGYPT', font('inter', 25), CREAM)
        snowfall(d, t, 46)
    elif kind == 'versus':
        tower(d, 270, ground); temple(d, 790, ground)
        for i, (x, yy, lab, sub_) in enumerate(((270, ground - 620, '1096', 'OXFORD'), (790, ground - 500, '1325', 'AZTEC CAPITAL'))):
            p = back((t - 0.5 - i * 1.3) / 0.4)
            if p <= 0.02: continue
            col = POP[i * 2 % len(POP)]
            text_c(d, (x, yy), lab, font('anton', 120 * p), col, stroke=7)
            if p > 0.8: text_c(d, (x, yy + 86), sub_, font('inter', 34), CREAM)
        sparkles(d, t, 14)
    elif kind == 'follow_time':
        sparkles(d, t)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 110 - bh / 2, cx + bw / 2, VY0 + 110 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 110), 'FOLLOW', font('anton', 96 * p), CREAM)
        hourglass(d, cx, VY0 + 460, 110, 160, t, period=3.0)
    elif kind == 'follow':
        p = 1 + 0.06 * math.sin(t * 6)
        bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 280 - bh / 2, cx + bw / 2, VY0 + 280 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 280), 'FOLLOW', font('anton', 96 * p), CREAM)
        emu(d, cx - 300, ground, 0.9, t, face=1); emu(d, cx + 300, ground, 0.9, t + 0.5, face=-1)

_stamp = {}
def stamp(word, p):
    PW, PH = int(660 * S), int(220 * S)
    if word not in _stamp:
        im = Image.new('RGBA', (PW, PH), (0, 0, 0, 0)); d = SD(ImageDraw.Draw(im))
        d.rounded_rectangle([10, 10, 650, 210], 18, outline=RED, width=14)
        text_c(d, (330, 112), word, font('anton', 140), RED)
        _stamp[word] = im.rotate(-8, resample=Image.BICUBIC)
    im = _stamp[word]
    s = max(0.05, 1.8 - 0.8 * p)
    big = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))))
    out = Image.new('RGBA', (PW, PH), (0, 0, 0, 0))
    if s >= 1:
        x0, y0 = (big.width - PW) // 2, (big.height - PH) // 2
        out.alpha_composite(big.crop((x0, y0, x0 + PW, y0 + PH)))
    else:
        out.alpha_composite(big, ((PW - big.width) // 2, (PH - big.height) // 2))
    return out

def count_text(big, p):
    m = re.fullmatch(r'([\d,]+)(.*)', big)
    if not m: return big
    n = int(m.group(1).replace(',', ''))
    if n < 10: return big                      # tiny numbers look odd counting up
    if ',' not in m.group(1) and 1000 <= n <= 2100 and not m.group(2): return big   # a year, not a quantity
    v = int(round(n * ease(p)))
    return (f'{v:,}' if ',' in m.group(1) else str(v)) + m.group(2)

def chunks(d, words):
    """Split words into single-line groups that fit the screen width."""
    maxw = LAYOUT['cw']
    f = font('anton', LAYOUT['cap']); out, cur = [], []
    for i, w in enumerate(words):
        test = ' '.join(words[j].upper() for j in cur + [i])
        if cur and (d.textlength(test, font=f) > maxw or len(cur) >= LAYOUT['cn']):
            out.append(cur); cur = []
        cur.append(i)
    return out + [cur]

def caption(d, words, times, t):
    """Show the current line of words, current word highlighted."""
    cur = max([i for i, (a, _) in enumerate(times) if a <= t] or [0])
    chunk = next(c for c in chunks(d, words) if cur in c)
    f = font('anton', LAYOUT['cap'])
    widths = [d.textlength(words[i].upper() + ' ', font=f) for i in chunk]
    x = LAYOUT['cx'] - (sum(widths) - d.textlength(' ', font=f)) / 2; y = LAYOUT['cy']
    for i, w in zip(chunk, widths):
        d.text((x, y), words[i].upper(), font=f, fill=GOLD if i == cur else CREAM, anchor='lm', stroke_width=7, stroke_fill=INK)
        x += w

# ---------- main ----------
def compose(base, sc, lt, frac):
    """Draw one frame: the scene's visual, kicker, headline, sub line, caption and progress bar."""
    img = base.copy(); raw = ImageDraw.Draw(img); d = SD(raw)
    L = LAYOUT; tx = L['tx']
    visual(SD(raw, L['vox'], L['voy']), sc['visual'], lt, sc['dur'], img, sc)
    # kicker pill
    if sc['kicker']:
        f = font('inter', 40); tw = d.textlength(sc['kicker'], font=f)
        a = ease(lt / 0.3)
        d.rounded_rectangle([tx - tw / 2 - 30, L['ky'] - 36, tx + tw / 2 + 30, L['ky'] + 36], 36, fill=RED + (int(255 * a),))
        text_c(d, (tx, L['ky']), sc['kicker'], f, CREAM + (int(255 * a),))
    # big headline: pops in, numbers count up
    p = back(lt / 0.45)
    big = count_text(sc['big'], lt / 0.9)
    size = 230 if len(sc['big']) <= 6 else (175 if len(sc['big']) <= 9 else 140)
    text_c(d, (tx, L['by']), big, font('anton', size * max(p, 0.05)), GOLD, stroke=8)
    text_c(d, (tx, L['sy']), sc['sub'], font('inter', 50 if len(sc['sub']) <= 30 else 40), CREAM + (int(255 * ease((lt - 0.2) / 0.4)),))
    if lt < sc['dur'] - GAP + 0.1:
        caption(d, sc['words'], sc['times'], lt)
    # progress bar
    d.rectangle([0, FH - 10, FW * frac, FH], fill=GOLD)
    return img


def main(script_path, out_path):
    spec = json.load(open(script_path))
    set_palette(spec.get('palette', {}))
    set_layout(spec.get('layout', 'tall'), spec)
    tmp = tempfile.mkdtemp()
    voice, speed = spec.get('voice', 'mb-us2'), spec.get('speed', 150)
    scenes, clips = [], []
    for i, sc in enumerate(spec['scenes']):
        p = os.path.join(tmp, f's{i}.wav'); tts(sc['say'], voice, speed, p)
        a = load_wav(p); clips.append(a)
        dur = len(a) / SR
        words = sc['say'].split()
        w = np.array([len(re.sub(r'\W', '', x)) + 2 for x in words], float)
        edges = np.concatenate([[0], np.cumsum(w) / w.sum() * dur])
        scenes.append(dict(sc, dur=dur + GAP, words=words, times=list(zip(edges[:-1], edges[1:]))))
    total = sum(s['dur'] for s in scenes) + 0.6
    n = int(total * SR)
    voice_track = np.zeros(n, np.float32); sfx = np.zeros(n, np.float32)
    t0 = 0.0; wh = whoosh()
    for s, a in zip(scenes, clips):
        k = int(t0 * SR); voice_track[k:k + len(a)] += a[:n - k]
        j = max(k - len(wh) // 2, 0); sfx[j:j + len(wh)] += wh[:n - j]
        s['t0'] = t0; t0 += s['dur']
    voice_track /= max(np.abs(voice_track).max(), 1e-6) / 0.9
    # clean up the voice on its own: cut rumble, lift the consonants, even out the level
    vin, vout = os.path.join(tmp, 'voice.f32'), os.path.join(tmp, 'voice_fx.f32')
    voice_track.tofile(vin)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', vin,
                    '-af', VOICE_FX_NEURAL if voice.startswith('kokoro:') else VOICE_FX_ROBOT,
                    '-f', 'f32le', '-ar', str(SR), '-ac', '1', vout], check=True)
    voice_track = np.fromfile(vout, np.float32)[:n]
    voice_track = np.pad(voice_track, (0, n - len(voice_track)))
    voice_track /= max(np.abs(voice_track).max(), 1e-6) / 0.95
    # duck the music while the voice is talking
    talk = np.clip(envelope(voice_track, int(0.25 * SR)) * 12, 0, 1)
    mix = voice_track + music_bed(n) * (1 - 0.7 * talk) + sfx
    wav = os.path.join(tmp, 'mix.f32')
    np.clip(mix, -1, 1).astype(np.float32).tofile(wav)

    bases = base_frames()
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{FW * S}x{FH * S}', '-r', str(FPS), '-i', '-',
                            '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', wav,
                            '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', str(LAYOUT['crf']),
                            *(['-maxrate', LAYOUT['maxrate'], '-bufsize', '12M'] if LAYOUT.get('maxrate') else []), '-profile:v', 'high', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest', '-movflags', '+faststart', out_path],
                           stdin=subprocess.PIPE)
    nf = int(total * FPS)
    for fi in range(nf):
        t = fi / FPS
        sc = next((s for s in scenes if s['t0'] <= t < s['t0'] + s['dur']), scenes[-1])
        lt = t - sc['t0']
        img = compose(bases[fi % len(bases)], sc, lt, t / total)
        # fade in/out of the whole video
        frame = img.convert('RGB')
        fade = min(1.0, t / 0.4, (total - t) / 0.5)
        if fade < 1: frame = Image.fromarray((np.asarray(frame, np.float32) * max(fade, 0)).astype(np.uint8))
        enc.stdin.write(frame.tobytes())
        if fi % (FPS * 5) == 0: print(f'  {t:5.1f}s / {total:.1f}s', flush=True)
    enc.stdin.close(); enc.wait()
    print(f'Done: {out_path} ({total:.1f}s)')

if __name__ == '__main__':
    args = sys.argv[1:]
    if '--scale' in args:                       # --scale 1 = 1080p, 2 = 4K (default); whole numbers only
        i = args.index('--scale'); S = int(args[i + 1]); del args[i:i + 2]
    main(args[0], args[1])
