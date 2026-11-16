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
    def __init__(self, d): self.d = d
    @staticmethod
    def p(xy): return [(v[0] * S, v[1] * S) if isinstance(v, (tuple, list)) else v * S for v in xy]
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
        self.d.text((xy[0] * S, xy[1] * S), s, font=font, fill=fill, anchor=anchor, stroke_width=round(stroke_width * S), stroke_fill=stroke_fill)
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
    t = min(max(t, 0.0), 1.0); c = 1.70158
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
    RW, RH = W * S, H * S
    y = np.linspace(0, 1, RH)[:, None, None]
    img = (np.array(BG0) * (1 - y) + np.array(BG1) * y) * np.ones((1, RW, 1))
    yy, xx = np.mgrid[0:RH, 0:RW]
    v = 1 - 0.55 * (((xx - RW / 2) / (RW * 0.75)) ** 2 + ((yy - RH / 2) / (RH * 0.62)) ** 2)
    img = (img * np.clip(v, 0.25, 1)[:, :, None]).astype(np.float32)
    if GLOW:                                    # soft coloured light behind the visual area
        gl = np.exp(-(((xx - RW / 2) / (RW * 0.48)) ** 2 + ((yy - RH * 0.53) / (RH * 0.2)) ** 2))
        img += (np.array(GLOW, np.float32) * 0.42) * gl[:, :, None].astype(np.float32)
    rng = np.random.default_rng(1); g = 2 * S
    out = []
    for _ in range(k):
        n = rng.normal(0, 7, (RH // g, RW // g, 1)).astype(np.float32).repeat(g, 0).repeat(g, 1)
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
            img.alpha_composite(stamp('JAMMED!', p), (int((cx - 330) * S), int((VY0 + 360) * S)))
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
            img.alpha_composite(stamp('SAFETY', back((t - 0.9) / 0.35)), (int((cx - 250) * S), int((VY0 + 330) * S)))
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
            text_c(d, (x, yy - hg + 50), '1518', font('inter', 40), INK)
            text_c(d, (x, yy + 50), mth, font('anton', 130), INK)
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
    v = int(round(n * ease(p)))
    return (f'{v:,}' if ',' in m.group(1) else str(v)) + m.group(2)

def chunks(d, words, maxw=930):
    """Split words into single-line groups that fit the screen width."""
    f = font('anton', 92); out, cur = [], []
    for i, w in enumerate(words):
        test = ' '.join(words[j].upper() for j in cur + [i])
        if cur and (d.textlength(test, font=f) > maxw or len(cur) >= 4):
            out.append(cur); cur = []
        cur.append(i)
    return out + [cur]

def caption(d, words, times, t):
    """Show the current line of words, current word highlighted."""
    cur = max([i for i, (a, _) in enumerate(times) if a <= t] or [0])
    chunk = next(c for c in chunks(d, words) if cur in c)
    f = font('anton', 92)
    widths = [d.textlength(words[i].upper() + ' ', font=f) for i in chunk]
    x = W / 2 - (sum(widths) - d.textlength(' ', font=f)) / 2; y = 1470
    for i, w in zip(chunk, widths):
        d.text((x, y), words[i].upper(), font=f, fill=GOLD if i == cur else CREAM, anchor='lm', stroke_width=7, stroke_fill=INK)
        x += w

# ---------- main ----------
def main(script_path, out_path):
    spec = json.load(open(script_path))
    set_palette(spec.get('palette', {}))
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
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W * S}x{H * S}', '-r', str(FPS), '-i', '-',
                            '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', wav,
                            '-af', 'loudnorm=I=-14:TP=-1.5:LRA=11',
                            '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
                            '-c:a', 'aac', '-b:a', '256k', '-ar', '48000', '-shortest', '-movflags', '+faststart', out_path],
                           stdin=subprocess.PIPE)
    nf = int(total * FPS)
    for fi in range(nf):
        t = fi / FPS
        sc = next((s for s in scenes if s['t0'] <= t < s['t0'] + s['dur']), scenes[-1])
        lt = t - sc['t0']
        img = bases[fi % len(bases)].copy(); d = SD(ImageDraw.Draw(img))
        visual(d, sc['visual'], lt, sc['dur'], img, sc)
        # kicker pill
        if sc['kicker']:
            f = font('inter', 40); tw = d.textlength(sc['kicker'], font=f)
            a = ease(lt / 0.3)
            d.rounded_rectangle([W / 2 - tw / 2 - 30, 190 - 36, W / 2 + tw / 2 + 30, 190 + 36], 36, fill=RED + (int(255 * a),))
            text_c(d, (W / 2, 190), sc['kicker'], f, CREAM + (int(255 * a),))
        # big headline: pops in, numbers count up
        p = back(lt / 0.45)
        big = count_text(sc['big'], lt / 0.9)
        size = 230 if len(sc['big']) <= 6 else 175
        text_c(d, (W / 2, 390), big, font('anton', size * max(p, 0.05)), GOLD, stroke=8)
        text_c(d, (W / 2, 560), sc['sub'], font('inter', 50), CREAM + (int(255 * ease((lt - 0.2) / 0.4)),))
        if lt < sc['dur'] - GAP + 0.1:
            caption(d, sc['words'], sc['times'], lt)
        # progress bar
        d.rectangle([0, H - 10, W * t / total, H], fill=GOLD)
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
