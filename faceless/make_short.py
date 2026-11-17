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

DEFAULT_PAL = dict(bg0=BG0, bg1=BG1, text=CREAM, headline=GOLD, accent=RED, ink=INK, line=LINE, liquid=LIQ, steel=STEEL, pop=POP)

def scene_palette(spec, sc):
    """The full palette for a scene: defaults, then the script's palette, then the scene's own."""
    return {**DEFAULT_PAL, **spec.get('palette', {}), **sc.get('palette', {})}

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

# ----- silhouette drawings for the jungle soldier story -----
SIL = (14, 8, 22)

def sun_disc(d, x, y, r, col=(255, 214, 120)):
    d.ellipse([x - r * 1.18, y - r * 1.18, x + r * 1.18, y + r * 1.18], fill=mix(BG1, col, 0.35))
    d.ellipse([x - r, y - r, x + r, y + r], fill=col)

def palm(d, x, ground, s, lean=0.3, t=0.0, col=None):
    col = col or SIL
    pts = [(x + lean * (j / 6) ** 2 * 260 * s, ground - j * 78 * s) for j in range(7)]
    d.line(pts, fill=col, width=max(3, int(24 * s)), joint='curve')
    xt, yt = pts[-1]
    for i, a in enumerate((-172, -140, -108, -74, -40, -8, 24, 156)):
        a = math.radians(a + math.sin(t * 1.3 + i) * 4); L = 250 * s
        fr = [(xt + math.cos(a) * L * k / 6, yt + math.sin(a) * L * k / 6 + (k / 6) ** 2 * 120 * s) for k in range(7)]
        d.line(fr, fill=col, width=max(2, int(15 * s)), joint='curve')
        for k in range(1, 7):                                           # leaflets hanging off the frond
            fx, fy = fr[k]; d.line([(fx, fy), (fx - math.cos(a) * 14 * s, fy + 44 * s * (1 - k / 9))], fill=col, width=max(2, int(7 * s)))

def jungle_floor(d, ground, col=None, seed=5):
    col = col or SIL; rng = random.Random(seed)
    pts = [(0, ground + 44)] + [(x, ground - 14 - rng.uniform(0, 26)) for x in range(0, W + 1, 54)] + [(W, ground + 44)]
    d.polygon(pts, fill=col)
    for i in range(60):
        gx = rng.uniform(0, W); h = rng.uniform(26, 74)
        d.polygon([(gx - 7, ground - 12), (gx + rng.uniform(-18, 18), ground - 12 - h), (gx + 7, ground - 12)], fill=col)

def soldier(d, x, ground, s, pose='stand', f=1, q=1.0, col=None, rifle=True, cap=True, pack=True):
    """A silhouette soldier, feet at (x, ground). f = 1 faces right, -1 left. q < 1 crouches."""
    col = col or SIL; Y = lambda v: ground - v * s * q
    for sg in (-1, 1): d.line([(x, Y(160)), (x + sg * (30 + (1 - q) * 70) * s, Y(78)), (x + sg * 30 * s, ground)], fill=col, width=max(3, int(30 * s)), joint='curve')
    if pack: d.rounded_rectangle([x - f * 70 * s if f > 0 else x + 30 * s, Y(290), x - f * 30 * s if f > 0 else x + 70 * s, Y(170)], 10 * s, fill=col)
    d.rounded_rectangle([x - 38 * s, Y(300), x + 38 * s, Y(140)], 16 * s, fill=col)
    hy = Y(342); d.ellipse([x - 34 * s, hy - 34 * s, x + 34 * s, hy + 34 * s], fill=col)
    if cap:
        d.pieslice([x - 38 * s, hy - 46 * s, x + 38 * s, hy + 20 * s], 180, 360, fill=col)
        d.rectangle([x, hy - 16 * s, x + f * 58 * s, hy - 6 * s] if f > 0 else [x - 58 * s, hy - 16 * s, x, hy - 6 * s], fill=col)
        d.polygon([(x - f * 34 * s, hy - 10 * s), (x - f * 44 * s, hy + 40 * s), (x - f * 20 * s, hy + 30 * s)], fill=col)
    sh = (x + f * 30 * s, Y(280))
    if pose == 'salute': d.line([sh, (x + f * 84 * s, Y(300)), (x + f * 34 * s, hy - 8 * s)], fill=col, width=max(3, int(20 * s)), joint='curve')
    elif pose == 'point': d.line([sh, (x + f * 150 * s, Y(300))], fill=col, width=max(3, int(20 * s)))
    elif pose == 'offer': d.line([sh, (x + f * 110 * s, Y(230))], fill=col, width=max(3, int(20 * s)))
    else: d.line([sh, (x + f * 44 * s, Y(170))], fill=col, width=max(3, int(20 * s)))
    if rifle: d.line([(x - f * 14 * s, Y(120)), (x - f * 60 * s, Y(440))], fill=col, width=max(3, int(11 * s)))

def paper(d, x, y, w, h, ang, col):
    c, s_ = math.cos(ang), math.sin(ang)
    d.polygon([(x + (dx * c - dy * s_), y + (dx * s_ + dy * c)) for dx, dy in ((-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2))], fill=col)

# ----- channel trailer drawings -----
def cabinet(d, cx, ground, t, open_k, labels):
    """A tall archive cabinet; the middle drawer slides out and labelled files rise from it."""
    w, h = 380, 440; x0, x1, top = cx - w / 2, cx + w / 2, ground - h
    wood, dark = shade(GOLD, 0.55), shade(GOLD, 0.38)
    d.rounded_rectangle([x0, top, x1, ground], 18, fill=wood)
    d.rectangle([x1 - 60, top + 10, x1 - 10, ground], fill=dark)
    for i in range(3):
        y0 = top + 24 + i * 138; off = open_k * 150 if i == 1 else 0
        d.rounded_rectangle([x0 + 24 + off * 0.1, y0, x1 - 24 + off * 0.1, y0 + 116], 12, fill=shade(GOLD, 0.7))
        d.rounded_rectangle([cx - 60, y0 + 46, cx + 60, y0 + 70], 12, fill=CREAM)
        if i == 1 and open_k > 0:                                       # drawer front pulled toward us
            d.rounded_rectangle([x0 - 26 * open_k, y0 + 116, x1 + 26 * open_k, y0 + 116 + 60 * open_k], 10, fill=shade(GOLD, 0.82))
    for j, lab in enumerate(labels):
        p = back((t - 0.6 - j * 0.32) / 0.45)
        if p <= 0.02: continue
        a = math.radians(-168 + j * (156 / max(1, len(labels) - 1)))
        fx = cx + math.cos(a) * 400 * min(p, 1.05); fy = top + 60 + math.sin(a) * 150 * min(p, 1.05) + math.sin(t * 2 + j) * 10
        col = POP[j % len(POP)]
        d.rounded_rectangle([fx - 84, fy - 54, fx + 84, fy + 54], 12, fill=col)
        d.rounded_rectangle([fx - 84, fy - 74, fx - 18, fy - 44], 10, fill=col)
        text_c(d, (fx, fy + 4), lab, font('anton', 58), INK)

def mini_icon(d, kind, x, y, s, t):
    if kind == 'emu': emu(d, x, y + 60 * s, 0.38 * s, t, face=1)
    elif kind == 'dancer': dancer(d, x, y + 70 * s, 0.42 * s, t * 1.4, POP[0], style=1)
    elif kind == 'palm': palm(d, x, y + 80 * s, 0.32 * s, 0.3, t, col=INK)
    elif kind == 'volcano':
        ty = volcano(d, x, y + 60 * s, w=170 * s, h=120 * s); d.ellipse([x - 22 * s, ty - 50 * s, x + 22 * s, ty - 6 * s], fill=FIRE)
    elif kind == 'wave': wave(d, x + 70 * s, y + 60 * s, 90 * s, t, left=x - 90 * s)
    elif kind == 'pyramid': pyramid(d, x, y + 60 * s, 170 * s, 120 * s)

# ----- ocean liner drawings (Violet Jessop story) -----
LIGHT, FUNNEL = (255, 222, 140), (214, 170, 96)

def liner(d, x, wl, s, tilt=0.0, hull=None, lights=True, funnels=4, band=None, cross=False, deck=None):
    """An ocean liner side view, bow to the right, waterline centre at (x, wl). tilt > 0 dips the bow."""
    hull = hull or INK; deck = deck or mix(hull, CREAM, 0.25)
    c, sn = math.cos(tilt), math.sin(tilt)
    R = lambda px, py: (x + (px * c - py * sn) * s, wl + (px * sn + py * c) * s)
    poly = lambda pts, col: d.polygon([R(px, py) for px, py in pts], fill=col)
    poly([(-330, 0), (320, 0), (385, -112), (-345, -112), (-362, -64)], hull)
    if band: poly([(-340, -40), (355, -40), (364, -56), (-350, -56)], band)
    poly([(-300, -112), (300, -112), (300, -150), (-300, -150)], deck)
    poly([(-240, -150), (220, -150), (220, -186), (-240, -186)], mix(deck, CREAM, 0.3))
    for i in range(funnels):
        fx = -170 + i * (360 / max(1, funnels - 1)) if funnels > 1 else 0
        poly([(fx - 24, -186), (fx + 24, -186), (fx + 30, -318), (fx - 18, -318)], FUNNEL)
        poly([(fx - 20, -296), (fx + 29, -296), (fx + 30, -318), (fx - 18, -318)], INK)
    d.line([R(-300, -150), R(-280, -380)], fill=mix(hull, CREAM, 0.4), width=max(2, int(5 * s)))
    d.line([R(280, -150), R(300, -380)], fill=mix(hull, CREAM, 0.4), width=max(2, int(5 * s)))
    if cross:
        for cxp in (-170, 160):
            poly([(cxp - 10, -96), (cxp + 10, -96), (cxp + 10, -22), (cxp - 10, -22)], (214, 40, 46))
            poly([(cxp - 36, -69), (cxp + 36, -69), (cxp + 36, -49), (cxp - 36, -49)], (214, 40, 46))
    if lights:
        for row in (-82, -130, -168):
            for i in range(-10, 11):
                px = i * 29
                if abs(px) > (300 if row < -100 else 320) or (row == -168 and abs(px + 10) > 215): continue
                if (i * 7 + row) % 5 == 0: continue
                cx_, cy_ = R(px, row); r = 3.5 * s
                d.ellipse([cx_ - r, cy_ - r, cx_ + r, cy_ + r], fill=LIGHT)

def warship(d, x, wl, s):
    col = STEEL; R = lambda px, py: (x + px * s, wl + py * s)
    d.polygon([R(-280, 0), R(230, 0), R(300, 18), R(262, -88), R(-292, -88)], fill=col)
    d.polygon([R(-160, -88), R(120, -88), R(110, -140), R(-150, -140)], fill=shade(col, 1.15))
    for fx in (-90, 10): d.polygon([R(fx - 20, -140), R(fx + 20, -140), R(fx + 18, -250), R(fx - 18, -250)], fill=shade(col, 0.8))
    d.line([R(80, -140), R(90, -330)], fill=shade(col, 0.8), width=max(2, int(6 * s)))

def water(d, y, t, dark=False):
    """The sea from y down, with moving wave lines."""
    d.rectangle([0, y, W, VY1 + 60], fill=shade(LIQ, 0.6) if dark else LIQ)
    for i in range(7):
        yy = y + 16 + i * 30; off = math.sin(t * 1.3 + i * 1.7) * 40
        for k in range(5):
            x0 = (k * 260 + off + i * 90) % (W + 200) - 120
            d.line([(x0, yy), (x0 + 120, yy)], fill=mix(LIQ, CREAM, 0.25 - i * 0.025), width=5)

def moon(d, x, y, r):
    d.ellipse([x - r * 1.6, y - r * 1.6, x + r * 1.6, y + r * 1.6], fill=mix(BG1, CREAM, 0.12))
    d.ellipse([x - r, y - r, x + r, y + r], fill=CREAM)
    d.ellipse([x - r * 0.4, y - r * 0.3, x - r * 0.1, y], fill=mix(CREAM, STEEL, 0.25))

def iceberg(d, x, wl, s):
    ice, sh = (226, 242, 255), (150, 196, 236)
    d.polygon([(x - 160 * s, wl), (x - 110 * s, wl - 150 * s), (x - 40 * s, wl - 230 * s), (x + 10 * s, wl - 330 * s), (x + 70 * s, wl - 210 * s), (x + 150 * s, wl - 120 * s), (x + 190 * s, wl)], fill=ice)
    d.polygon([(x + 10 * s, wl - 330 * s), (x + 70 * s, wl - 210 * s), (x + 150 * s, wl - 120 * s), (x + 190 * s, wl), (x + 40 * s, wl)], fill=sh)

def lifeboat(d, x, wl, s, t, baby=True):
    b = math.sin(t * 1.8) * 6 * s; r = math.sin(t * 1.4) * 0.04
    for i, px in enumerate((-110, -55, 0, 55, 110)):
        hx = x + px * s; hy = wl - 70 * s + b + px * r * s
        d.ellipse([hx - 18 * s, hy - 52 * s, hx + 18 * s, hy - 16 * s], fill=INK)
        d.rounded_rectangle([hx - 22 * s, hy - 18 * s, hx + 22 * s, hy + 30 * s], 10 * s, fill=INK)
        if baby and i == 2: d.ellipse([hx - 30 * s, hy - 8 * s, hx + 6 * s, hy + 16 * s], fill=CREAM)
    d.polygon([(x - 190 * s, wl - 60 * s + b), (x + 190 * s, wl - 60 * s + b), (x + 150 * s, wl + 6 * s + b), (x - 150 * s, wl + 6 * s + b)], fill=(150, 104, 62))
    d.line([(x - 190 * s, wl - 60 * s + b), (x + 190 * s, wl - 60 * s + b)], fill=(190, 140, 90), width=max(2, int(8 * s)))
    d.line([(x + 40 * s, wl - 50 * s + b), (x + 220 * s, wl + 30 * s + b)], fill=(130, 90, 54), width=max(2, int(8 * s)))

def propeller(d, cx, cy, r, ang):
    for k in range(3):
        a = ang + k * 2 * math.pi / 3
        p1 = (cx + math.cos(a - 0.35) * r * 0.25, cy + math.sin(a - 0.35) * r * 0.25)
        p2 = (cx + math.cos(a - 0.2) * r, cy + math.sin(a - 0.2) * r)
        p3 = (cx + math.cos(a + 0.25) * r * 0.95, cy + math.sin(a + 0.25) * r * 0.95)
        p4 = (cx + math.cos(a + 0.4) * r * 0.25, cy + math.sin(a + 0.4) * r * 0.25)
        d.polygon([p1, p2, p3, p4], fill=FUNNEL)
    d.ellipse([cx - r * 0.2, cy - r * 0.2, cx + r * 0.2, cy + r * 0.2], fill=shade(FUNNEL, 0.7))

def bubbles(d, t, x0, x1, y0, y1, n=24, seed=3):
    rng = random.Random(seed)
    for i in range(n):
        x = rng.uniform(x0, x1) + math.sin(t * 2 + i) * 14; y = y1 - (t * rng.uniform(60, 140) + rng.uniform(0, y1 - y0)) % (y1 - y0)
        r = rng.uniform(5, 16); d.ellipse([x - r, y - r, x + r, y + r], outline=mix(LIQ, CREAM, 0.5), width=3)

# ----- "on this day" drawings -----
def tear_calendar(d, cx, cy, s, month, day, sub='', tear=0.0, nxt=None):
    """A tear-off desk calendar. tear 0..1 peels the top page off to reveal nxt."""
    w, h = 300 * s, 360 * s
    d.rounded_rectangle([cx - w + 14 * s, cy - h + 20 * s, cx + w + 14 * s, cy + h + 20 * s], 26 * s, fill=shade(BG0, 0.7))
    def page(dd, num, col=CREAM):
        d.rounded_rectangle([cx - w, cy - h, cx + w, cy + h], 26 * s, fill=col)
        d.rounded_rectangle([cx - w, cy - h, cx + w, cy - h + 150 * s], 26 * s, fill=RED); d.rectangle([cx - w, cy - h + 100 * s, cx + w, cy - h + 150 * s], fill=RED)
        text_c(d, (cx, cy - h + 78 * s), month, font('anton', 92 * s), CREAM)
        text_c(d, (cx, cy + 50 * s), str(num), font('anton', 330 * s), INK)
        if sub: text_c(d, (cx, cy + h - 56 * s), sub, font('inter', 40 * s), shade(INK, 2.4))
    page(d, nxt if (tear > 0 and nxt is not None) else day)
    if 0 < tear < 1 and nxt is not None:                                   # the old page falls away
        k = ease(tear); pw, ph = w * (1 - 0.45 * k), h * (1 - 0.45 * k)
        px, py = cx + 420 * k * s, cy + 200 * k * s; a = 0.9 * k
        c_, s_ = math.cos(a), math.sin(a)
        pts = [(px + (dx * c_ - dy * s_), py + (dx * s_ + dy * c_)) for dx, dy in ((-pw, -ph), (pw, -ph), (pw, ph), (-pw, ph))]
        d.polygon(pts, fill=mix(CREAM, BG1, 0.25 * k))
        mx, my = px - (0 * c_ - 40 * s_), py + (0 * s_ + 40 * c_)
        text_c(d, (mx, my), str(day), font('anton', 300 * s * (1 - 0.45 * k)), mix(INK, BG1, 0.5 * k))
    for rx in (-150, 0, 150):                                               # binder rings
        d.rounded_rectangle([cx + rx * s - 14 * s, cy - h - 40 * s, cx + rx * s + 14 * s, cy - h + 30 * s], 12 * s, fill=STEEL)

def crown(d, cx, cy, s, t):
    g, dk = GOLD, shade(GOLD, 0.72)
    for i in range(12):                                                     # light rays
        a = i * math.pi / 6 + t * 0.25
        d.polygon([(cx, cy), (cx + math.cos(a - 0.08) * 420 * s, cy + math.sin(a - 0.08) * 420 * s), (cx + math.cos(a + 0.08) * 420 * s, cy + math.sin(a + 0.08) * 420 * s)], fill=mix(BG1, GOLD, 0.18))
    d.rounded_rectangle([cx - 260 * s, cy + 120 * s, cx + 260 * s, cy + 190 * s], 30 * s, fill=RED)        # cushion
    d.polygon([(cx - 210 * s, cy + 120 * s), (cx - 240 * s, cy - 130 * s), (cx - 120 * s, cy - 10 * s), (cx, cy - 200 * s), (cx + 120 * s, cy - 10 * s), (cx + 240 * s, cy - 130 * s), (cx + 210 * s, cy + 120 * s)], fill=g)
    d.rectangle([cx - 210 * s, cy + 50 * s, cx + 210 * s, cy + 120 * s], fill=dk)
    for jx, col in ((-120, RED), (0, (60, 140, 230)), (120, RED)): d.ellipse([cx + jx * s - 22 * s, cy + 63 * s, cx + jx * s + 22 * s, cy + 107 * s], fill=col)
    for tx, ty in ((-240, -130), (0, -200), (240, -130)): d.ellipse([tx * s + cx - 22 * s, ty * s + cy - 22 * s, tx * s + cx + 22 * s, ty * s + cy + 22 * s], fill=CREAM)

def wooden_mouse(d, cx, ground, s, t):
    wd, dk = (176, 116, 62), (130, 82, 42)
    d.rounded_rectangle([cx - 200 * s, ground - 210 * s, cx + 200 * s, ground], 30 * s, fill=wd)
    for i in range(5): d.line([(cx - 180 * s, ground - (40 + i * 34) * s), (cx + 180 * s, ground - (46 + i * 34) * s)], fill=dk, width=max(2, int(4 * s)))
    d.rounded_rectangle([cx + 70 * s, ground - 250 * s, cx + 150 * s, ground - 206 * s], 10 * s, fill=RED)   # the button
    d.ellipse([cx - 150 * s, ground - 30 * s, cx - 90 * s, ground + 30 * s], fill=INK); d.ellipse([cx + 90 * s, ground - 30 * s, cx + 150 * s, ground + 30 * s], fill=INK)
    cord = [(cx - 200 * s, ground - 120 * s), (cx - 290 * s, ground - 150 * s), (cx - 330 * s, ground - 300 * s), (cx - 300 * s, ground - 440 * s)]
    d.line(cord, fill=INK, width=max(3, int(12 * s)), joint='curve')

def monitor(d, cx, cy, s, t):
    d.rounded_rectangle([cx - 220 * s, cy - 170 * s, cx + 220 * s, cy + 150 * s], 30 * s, fill=(196, 186, 160))
    d.rounded_rectangle([cx - 180 * s, cy - 136 * s, cx + 180 * s, cy + 110 * s], 20 * s, fill=(20, 40, 30))
    for i in range(4): d.line([(cx - 150 * s, cy - 100 * s + i * 40 * s), (cx + (60 if i % 2 else 120) * s, cy - 100 * s + i * 40 * s)], fill=(80, 220, 120), width=max(2, int(6 * s)))
    ax, ay = cx + math.sin(t * 1.3) * 110 * s, cy + math.cos(t * 1.7) * 60 * s                 # cursor
    d.polygon([(ax, ay), (ax, ay + 50 * s), (ax + 14 * s, ay + 38 * s), (ax + 26 * s, ay + 60 * s), (ax + 36 * s, ay + 54 * s), (ax + 24 * s, ay + 34 * s), (ax + 40 * s, ay + 34 * s)], fill=CREAM)
    d.rectangle([cx - 60 * s, cy + 150 * s, cx + 60 * s, cy + 200 * s], fill=(170, 160, 136))

def lunar_rover(d, x, ground, s, t):
    body = (150, 156, 170)
    for wx in (-120, -40, 40, 120):
        r = 34 * s; d.ellipse([x + wx * s - r, ground - 2 * r, x + wx * s + r, ground], fill=INK)
        a = t * 3 + wx; d.line([(x + wx * s, ground - r), (x + wx * s + math.cos(a) * r, ground - r + math.sin(a) * r)], fill=STEEL, width=max(2, int(5 * s)))
    d.pieslice([x - 170 * s, ground - 210 * s, x + 170 * s, ground - 30 * s], 0, 180, fill=body)
    d.rectangle([x - 170 * s, ground - 124 * s, x + 170 * s, ground - 110 * s], fill=shade(body, 0.7))
    d.polygon([(x - 170 * s, ground - 124 * s), (x - 200 * s, ground - 230 * s), (x + 140 * s, ground - 300 * s), (x + 170 * s, ground - 124 * s)], fill=(60, 90, 160))  # open lid / solar panel
    for i in range(1, 5): d.line([(x - 170 * s + i * 66 * s, ground - 124 * s), (x - 200 * s + i * 68 * s, ground - 230 * s - i * 14 * s)], fill=(40, 60, 110), width=max(2, int(4 * s)))
    d.line([(x + 120 * s, ground - 140 * s), (x + 150 * s, ground - 360 * s)], fill=STEEL, width=max(2, int(5 * s)))
    d.ellipse([x + 186 * s, ground - 150 * s, x + 214 * s, ground - 122 * s], fill=LIGHT)

def candle(d, x, y, s, t, i=0):
    d.rectangle([x - 7 * s, y - 40 * s, x + 7 * s, y], fill=CREAM)
    fk = 1 + 0.25 * math.sin(t * 13 + i * 2)
    d.ellipse([x - 9 * s, y - 70 * s * fk, x + 9 * s, y - 38 * s], fill=HOT)

# ----- Apollo 11 drawings -----
MOONC = (176, 172, 166)

def moon_ground(d, ground, top=150):
    d.polygon([(x_, ground - top + 850 * (1 - math.sqrt(max(0, 1 - ((x_ - W / 2) / (W / 2 + 260)) ** 2)))) for x_ in range(-260, W + 261, 30)] + [(W + 260, ground + 40), (-260, ground + 40)], fill=MOONC)
    for cx_, cy_, r in ((160, ground - 40, 60), (700, ground - 90, 40), (930, ground - 20, 80), (420, ground - 10, 34)):
        d.ellipse([cx_ - r, cy_ - r * 0.36, cx_ + r, cy_ + r * 0.36], fill=shade(MOONC, 0.75))

def earth(d, x, y, r):
    d.ellipse([x - r, y - r, x + r, y + r], fill=(44, 110, 190))
    for bx, by, bw, bh in ((-0.3, -0.3, 0.7, 0.55), (0.35, 0.25, 0.55, 0.65), (-0.4, 0.4, 0.45, 0.3)):
        d.ellipse([x + bx * r - bw * r / 2, y + by * r - bh * r / 2, x + bx * r + bw * r / 2, y + by * r + bh * r / 2], fill=(110, 180, 110))
    d.ellipse([x - r * 0.5, y - r * 0.85, x + r * 0.2, y - r * 0.6], fill=(236, 244, 255))

def big_moon(d, x, y, r):
    d.ellipse([x - r, y - r, x + r, y + r], fill=MOONC)
    for bx, by, br in ((-0.35, -0.2, 0.22), (0.25, 0.3, 0.16), (0.3, -0.4, 0.12), (-0.1, 0.5, 0.1), (0.5, 0.0, 0.08)):
        d.ellipse([x + bx * r - br * r, y + by * r - br * r, x + bx * r + br * r, y + by * r + br * r], fill=shade(MOONC, 0.8))

def saturn_v(d, x, base, s, flame=0.0, t=0.0):
    W_, B = (240, 242, 248), INK
    if flame > 0:
        fk = 1 + 0.15 * math.sin(t * 30)
        d.polygon([(x - 46 * s, base), (x, base + 300 * s * flame * fk), (x + 46 * s, base)], fill=(255, 150, 40))
        d.polygon([(x - 26 * s, base), (x, base + 180 * s * flame * fk), (x + 26 * s, base)], fill=(255, 236, 160))
    d.rectangle([x - 50 * s, base - 300 * s, x + 50 * s, base], fill=W_)
    for yy in (-60, -150, -240): d.rectangle([x - 50 * s, base + (yy - 18) * s, x + 50 * s, base + yy * s], fill=B)
    for sg in (-1, 1): d.polygon([(x + sg * 50 * s, base - 60 * s), (x + sg * 86 * s, base), (x + sg * 50 * s, base)], fill=B)
    d.polygon([(x - 50 * s, base - 300 * s), (x - 38 * s, base - 350 * s), (x + 38 * s, base - 350 * s), (x + 50 * s, base - 300 * s)], fill=W_)
    d.rectangle([x - 38 * s, base - 470 * s, x + 38 * s, base - 350 * s], fill=W_); d.rectangle([x - 38 * s, base - 410 * s, x + 38 * s, base - 396 * s], fill=B)
    d.polygon([(x - 38 * s, base - 470 * s), (x - 22 * s, base - 520 * s), (x + 22 * s, base - 520 * s), (x + 38 * s, base - 470 * s)], fill=W_)
    d.polygon([(x - 22 * s, base - 520 * s), (x, base - 580 * s), (x + 22 * s, base - 520 * s)], fill=(200, 204, 214))
    d.line([(x, base - 580 * s), (x, base - 630 * s)], fill=(200, 204, 214), width=max(2, int(5 * s)))

def puff_cloud(d, x, y, t, n=16, spread=380, col=(220, 222, 230)):
    rng = random.Random(14)
    for i in range(n):
        a = rng.uniform(math.pi * 0.95, math.pi * 2.05); age = (t * 0.6 + i / n) % 1
        r = 40 + age * 90; px = x + math.cos(a) * spread * age; py = y + math.sin(a) * 60 * age - 20
        d.ellipse([px - r, py - r * 0.7, px + r, py + r * 0.7], fill=mix(col, BG1, age * 0.6))

def ascent_stage(d, x, base, s):
    c = (200, 204, 214)
    d.polygon([(x - 70 * s, base), (x - 80 * s, base - 70 * s), (x - 40 * s, base - 120 * s), (x + 50 * s, base - 120 * s), (x + 80 * s, base - 70 * s), (x + 70 * s, base)], fill=c)
    d.polygon([(x - 30 * s, base - 80 * s), (x - 6 * s, base - 100 * s), (x - 6 * s, base - 60 * s)], fill=INK)
    d.line([(x + 40 * s, base - 120 * s), (x + 60 * s, base - 180 * s)], fill=c, width=max(2, int(5 * s))); d.ellipse([x + 46 * s, base - 206 * s, x + 80 * s, base - 172 * s], fill=CREAM)

def descent_stage(d, x, ground, s):
    d.line([(x - 70 * s, ground - 70 * s), (x - 120 * s, ground)], fill=STEEL, width=max(3, int(9 * s))); d.line([(x + 70 * s, ground - 70 * s), (x + 120 * s, ground)], fill=STEEL, width=max(3, int(9 * s)))
    for fx in (-120, 120): d.ellipse([x + fx * s - 24 * s, ground - 8 * s, x + fx * s + 24 * s, ground + 8 * s], fill=STEEL)
    d.polygon([(x - 90 * s, ground - 140 * s), (x + 90 * s, ground - 140 * s), (x + 100 * s, ground - 66 * s), (x - 100 * s, ground - 66 * s)], fill=(214, 168, 62))
    for i in range(5): d.line([(x - 84 * s + i * 42 * s, ground - 136 * s), (x - 92 * s + i * 46 * s, ground - 70 * s)], fill=(180, 136, 40), width=max(2, int(3 * s)))

def capsule(d, x, y, s):
    d.polygon([(x - 70 * s, y), (x - 26 * s, y - 100 * s), (x + 26 * s, y - 100 * s), (x + 70 * s, y)], fill=(210, 214, 222))
    d.rounded_rectangle([x - 74 * s, y - 6 * s, x + 74 * s, y + 18 * s], 8 * s, fill=(120, 96, 70))
    d.ellipse([x - 12 * s, y - 70 * s, x + 12 * s, y - 46 * s], fill=INK)

def chutes(d, x, y, s, t):
    for i, dx in enumerate((-150, 0, 150)):
        cx_, cy_ = x + dx * s, y - 330 * s + abs(dx) * 0.25 * s + math.sin(t * 2 + i) * 6 * s
        d.line([(cx_ - 110 * s, cy_), (x, y - 100 * s)], fill=CREAM, width=max(1, int(3 * s))); d.line([(cx_ + 110 * s, cy_), (x, y - 100 * s)], fill=CREAM, width=max(1, int(3 * s)))
        for k in range(4):
            d.pieslice([cx_ - 110 * s, cy_ - 90 * s, cx_ + 110 * s, cy_ + 90 * s], 180 + k * 45, 180 + (k + 1) * 45, fill=(226, 60, 50) if k % 2 == 0 else (245, 245, 245))

def boot_print(d, x, y, s, a=0.0):
    c_, s_ = math.cos(a), math.sin(a); R = lambda px, py: (x + px * c_ * s - py * s_ * s, y + px * s_ * s + py * c_ * s)
    d.polygon([R(px, py) for px, py in ((-40, -160), (40, -160), (66, -110), (70, -20), (52, 40), (56, 130), (-56, 130), (-52, 40), (-70, -20), (-66, -110))], fill=shade(MOONC, 0.68))
    d.line([R(-56, 50), R(56, 50)], fill=shade(MOONC, 0.9), width=max(2, int(12 * s)))
    for k in range(9):
        yy = -130 + k * 30; d.line([R(-50, yy), R(50, yy)], fill=shade(MOONC, 0.52), width=max(2, int(8 * s)))

# ----- Zanzibar war drawings -----
WALL = (238, 232, 218)

def palace(d, cx, ground, s, damage=0.0, flag=1.0, t=0.0):
    """A white seafront palace with arcades and a central tower. damage 0..1 knocks it about."""
    w, h = 520 * s, 230 * s; x0, x1 = cx - w / 2, cx + w / 2
    d.rectangle([x0, ground - h, x1, ground], fill=WALL)
    d.rectangle([cx - 70 * s, ground - h - 170 * s, cx + 70 * s, ground - h], fill=WALL)
    d.polygon([(cx - 84 * s, ground - h - 170 * s), (cx, ground - h - 240 * s), (cx + 84 * s, ground - h - 170 * s)], fill=shade(WALL, 0.85))
    for row in range(2):
        for i in range(9):
            ax = x0 + 32 * s + i * 54 * s; ay = ground - h + 34 * s + row * 104 * s
            d.rectangle([ax, ay + 26 * s, ax + 30 * s, ay + 80 * s], fill=shade(WALL, 0.55))
            d.pieslice([ax, ay + 8 * s, ax + 30 * s, ay + 44 * s], 180, 360, fill=shade(WALL, 0.55))
    d.rectangle([x0, ground - h, x1, ground - h + 16 * s], fill=shade(WALL, 0.8))
    pole_top = ground - h - 240 * s - 150 * s
    d.line([(cx, ground - h - 240 * s), (cx, pole_top)], fill=INK, width=max(2, int(6 * s)))
    if flag > 0:
        fy = pole_top + (1 - flag) * 130 * s; wv = math.sin(t * 6) * 8 * s
        d.polygon([(cx, fy), (cx + 120 * s, fy + 10 * s + wv), (cx + 120 * s, fy + 70 * s + wv), (cx, fy + 60 * s)], fill=(206, 32, 42))
    if damage > 0:
        rng = random.Random(3)
        for i in range(int(14 * damage)):
            hx, hy = rng.uniform(x0 + 20 * s, x1 - 20 * s), rng.uniform(ground - h - 120 * s, ground - 30 * s)
            r = rng.uniform(18, 46) * s; d.ellipse([hx - r, hy - r * 0.8, hx + r, hy + r * 0.8], fill=shade(INK, 1.6))
        k = min(1, damage * 1.3)
        d.polygon([(x1 - 140 * s, ground - h), (x1 + 4, ground - h), (x1 + 4, ground - h + 160 * s * k), (x1 - 60 * s, ground - h + 60 * s * k)], fill=BG1 if False else shade(INK, 1.2))

def clock_face(d, cx, cy, r, hours, minutes, ring=None):
    d.ellipse([cx - r - 16, cy - r - 16, cx + r + 16, cy + r + 16], fill=ring or GOLD)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=CREAM)
    for i in range(12):
        a = math.radians(i * 30 - 90); L = 30 if i % 3 == 0 else 16
        d.line([(cx + math.cos(a) * (r - L - 8), cy + math.sin(a) * (r - L - 8)), (cx + math.cos(a) * (r - 8), cy + math.sin(a) * (r - 8))], fill=INK, width=8 if i % 3 == 0 else 5)
    ha = math.radians((hours % 12 + minutes / 60) * 30 - 90); ma = math.radians(minutes * 6 - 90)
    d.line([(cx, cy), (cx + math.cos(ha) * r * 0.52, cy + math.sin(ha) * r * 0.52)], fill=INK, width=16)
    d.line([(cx, cy), (cx + math.cos(ma) * r * 0.8, cy + math.sin(ma) * r * 0.8)], fill=RED, width=10)
    d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], fill=INK)

def shell_arcs(d, t, xs, x1, y1, n=6):
    """Shells flying from ships at xs (x, y) toward (x1, y1)."""
    for i in range(n):
        x0, y0 = xs[i % len(xs)]; u = (t * 0.9 + i / n) % 1
        x = x0 + (x1 - x0) * u; y = y0 + (y1 - y0) * u - math.sin(u * math.pi) * 260
        for k in range(4):
            uu = max(0, u - k * 0.03); xx = x0 + (x1 - x0) * uu; yy = y0 + (y1 - y0) * uu - math.sin(uu * math.pi) * 260
            r = 10 - k * 2; d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=mix(HOT, BG1, k * 0.25))
        if u < 0.08:
            r = 40 * (1 - u / 0.08); d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=HOT)

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
        sea(d, ground); ty = volcano(d, cx, ground, shake=sh, h=(sc or {}).get('h', 500))
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
        d.polygon([(x_, ground - 110 + 810 * (1 - math.sqrt(max(0, 1 - ((x_ - cx) / (cx + 260)) ** 2)))) for x_ in range(-260, W + 261, 30)] + [(W + 260, ground + 40), (-260, ground + 40)], fill=shade(STONE, 0.78))
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
    elif kind == 'jungle_soldier':
        sun_disc(d, cx, VY0 + 330, 250)
        palm(d, 120, ground, 1.05, 0.35, t); palm(d, 960, ground, 0.9, -0.3, t); palm(d, 330, ground, 0.6, -0.2, t)
        jungle_floor(d, ground)
        soldier(d, cx + 20, ground - 14, 1.25 + 0.02 * math.sin(t * 2))
    elif kind == 'island':
        sun_disc(d, 820, VY0 + 170, 110)
        for i in range(6):                                                      # waves
            yy = ground - 150 + i * 34; off = math.sin(t * 1.4 + i) * 30
            d.line([(60 + off, yy), (W - 60 + off, yy)], fill=mix(BG1, SIL, 0.35 + i * 0.08), width=8)
        pts = [(170, ground - 120), (300, ground - 250), (470, ground - 300), (640, ground - 262), (790, ground - 200), (900, ground - 120)]
        d.polygon(pts + [(900, ground - 100), (170, ground - 100)], fill=SIL)
        palm(d, 420, ground - 290, 0.55, 0.3, t); palm(d, 600, ground - 266, 0.45, -0.3, t)
        p = back((t - 0.7) / 0.4)
        if p > 0.02:
            px_, py_ = 520, ground - 420 - 60 * (1 - min(p, 1))
            d.ellipse([px_ - 44 * p, py_ - 44 * p, px_ + 44 * p, py_ + 44 * p], fill=RED); d.polygon([(px_ - 30 * p, py_ + 30 * p), (px_ + 30 * p, py_ + 30 * p), (px_, py_ + 100 * p)], fill=RED)
            d.ellipse([px_ - 16 * p, py_ - 16 * p, px_ + 16 * p, py_ + 16 * p], fill=CREAM)
    elif kind == 'orders':
        sun_disc(d, 230, VY0 + 300, 190); jungle_floor(d, ground)
        soldier(d, 250, ground - 14, 1.3, pose='point', rifle=False, pack=False)
        lines = (sc or {}).get('lines', ['NEVER', 'SURRENDER'])
        for i, ln in enumerate(lines):
            p = back((t - 0.6 - i * 0.7) / 0.4)
            if p <= 0.02: continue
            yy = VY0 + 190 + i * 170
            d.rounded_rectangle([cx - 20, yy - 70 * p, W - 50, yy + 70 * p], 18, fill=CREAM)
            text_c(d, ((cx - 20 + W - 50) / 2, yy), ln, font('anton', 86 * min(p, 1)), RED if i == len(lines) - 1 else SIL)
    elif kind == 'leaflets':
        sun_disc(d, 800, VY0 + 250, 150)
        px_ = (t * 260) % (W + 500) - 250; py_ = VY0 + 90                       # a plane crossing
        d.ellipse([px_ - 120, py_ - 20, px_ + 120, py_ + 20], fill=SIL); d.polygon([(px_ - 20, py_), (px_ - 70, py_ - 80), (px_ + 30, py_)], fill=SIL); d.polygon([(px_ - 20, py_), (px_ - 70, py_ + 80), (px_ + 30, py_)], fill=SIL)
        d.polygon([(px_ - 120, py_), (px_ - 150, py_ - 50), (px_ - 96, py_)], fill=SIL)
        palm(d, 130, ground, 1.0, 0.35, t); palm(d, 300, ground, 0.7, -0.2, t)
        jungle_floor(d, ground); soldier(d, 220, ground - 14, 1.0, q=0.62)
        rng = random.Random(12)
        for i in range(16):
            x = rng.uniform(380, W - 60) + math.sin(t * 2 + i) * 40
            y = VY0 + 120 + (t * rng.uniform(90, 150) + rng.uniform(0, 560)) % 560
            paper(d, x, y, 58, 76, t * rng.uniform(1, 2.4) + i, mix(CREAM, BG1, 0.1 + 0.25 * (i % 3) / 2))
    elif kind == 'years':
        k = ease(t / max(dur * 0.85, 1.5)); yr = 1945 + int(round(29 * k))
        a = (t * 1.6) % math.pi; sun_disc(d, cx - math.cos(a) * 460, VY0 + 420 - math.sin(a) * 330, 80)       # days racing by
        palm(d, 60, ground, 0.8, 0.2, t); palm(d, 1020, ground, 0.8, -0.2, t)
        jungle_floor(d, ground); soldier(d, cx - 30, ground - 14, 1.0, q=0.62)
        text_c(d, (cx, VY0 + 230), str(yr), font('anton', 300), CREAM, stroke=12)
    elif kind == 'alone':
        sun_disc(d, cx, VY0 + 330, 230); jungle_floor(d, ground)
        for i, x in enumerate((180, 420, 660, 900)):
            k = ease((t - 0.9 - (i if i < 1 else i - 1) * 0.9) / 0.7) if i != 1 else 0.0
            if k >= 1: continue
            soldier(d, x, ground - 14 + k * 30, 1.05, col=mix(SIL, mix(BG1, (255, 214, 120), 0.4), k))
    elif kind == 'found':
        sparkles(d, t, 18)
        palm(d, 980, ground, 1.0, -0.35, t); jungle_floor(d, ground)
        d.polygon([(70, ground - 10), (230, ground - 290), (390, ground - 10)], fill=SIL); d.polygon([(190, ground - 10), (230, ground - 150), (270, ground - 10)], fill=mix(BG1, (255, 214, 120), 0.5))
        fk = 1 + 0.2 * math.sin(t * 15) * math.sin(t * 4)                                             # campfire
        d.polygon([(cx - 46, ground - 14), (cx - 16, ground - 130 * fk), (cx, ground - 70), (cx + 22, ground - 160 * fk), (cx + 50, ground - 14)], fill=FIRE)
        d.polygon([(cx - 22, ground - 14), (cx + 4, ground - 84 * fk), (cx + 26, ground - 14)], fill=HOT)
        d.line([(cx - 70, ground - 8), (cx + 70, ground - 24)], fill=SIL, width=16); d.line([(cx - 70, ground - 24), (cx + 70, ground - 8)], fill=SIL, width=16)
        soldier(d, 380, ground - 14, 1.0, f=1, rifle=False, cap=False)
        soldier(d, min(900, 1200 - ease(t / 2.0) * 430), ground - 14, 1.05, f=-1)
    elif kind == 'relieved':
        sun_disc(d, cx, VY0 + 330, 250); jungle_floor(d, ground)
        soldier(d, 330, ground - 14, 1.2, pose='offer', f=1, rifle=False, cap=False, pack=False)
        d.rectangle([286, ground - 14 - 470, 374, ground - 14 - 452], fill=SIL); d.rectangle([302, ground - 14 - 510, 358, ground - 14 - 460], fill=SIL)      # a civilian hat
        b = math.sin(t * 2) * 4
        d.rounded_rectangle([430, ground - 370 + b, 560, ground - 210 + b], 8, fill=CREAM)
        for i in range(4): d.line([(448, ground - 340 + i * 34 + b), (542, ground - 340 + i * 34 + b)], fill=SIL, width=6)
        soldier(d, 760, ground - 14, 1.2, pose='salute' if t > 1.0 else 'stand', f=-1)
    elif kind == 'sword':
        sun_disc(d, cx, VY0 + 330, 270); jungle_floor(d, ground)
        soldier(d, 300, ground - 14, 1.2, pose='offer', f=1, rifle=False, cap=False, pack=False)
        d.rectangle([256, ground - 14 - 470, 344, ground - 14 - 452], fill=SIL); d.rectangle([272, ground - 14 - 510, 328, ground - 14 - 460], fill=SIL)
        soldier(d, 790, ground - 14, 1.2, pose='offer', f=-1, rifle=False)
        k = ease(t / 1.6); sx = 760 - 190 * k; sy_ = ground - 14 - 276
        d.line([(sx - 150, sy_ - 10), (sx + 90, sy_ + 4)], fill=CREAM, width=10); d.line([(sx + 90, sy_ + 4), (sx + 160, sy_ + 8)], fill=SIL, width=16); d.line([(sx + 86, sy_ - 22), (sx + 92, sy_ + 30)], fill=(255, 214, 120), width=8)
    elif kind == 'follow_jungle':
        sun_disc(d, cx, VY0 + 400, 200)
        palm(d, 120, ground, 1.0, 0.35, t); palm(d, 960, ground, 0.95, -0.3, t); jungle_floor(d, ground)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 110 - bh / 2, cx + bw / 2, VY0 + 110 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 110), 'FOLLOW', font('anton', 96 * p), CREAM)
        soldier(d, cx, ground - 14, 0.9, pose='salute')
    elif kind == 'archive':
        sparkles(d, t, 22)
        cabinet(d, cx, ground, t, ease((t - 0.2) / 0.6), (sc or {}).get('labels', ['1518', '1815', '1919', '1932', '1944']))
    elif kind == 'stopwatch':
        sparkles(d, t, 16); gx, gy, r = cx, VY0 + 360, 260
        d.rounded_rectangle([gx - 50, gy - r - 80, gx + 50, gy - r - 20], 16, fill=CREAM)
        d.ellipse([gx - r - 20, gy - r - 20, gx + r + 20, gy + r + 20], fill=CREAM); d.ellipse([gx - r, gy - r, gx + r, gy + r], fill=mix(BG1, INK, 0.5))
        k = min(1, t / max(dur * 0.85, 1))
        d.pieslice([gx - r + 14, gy - r + 14, gx + r - 14, gy + r - 14], -90, -90 + 360 * k, fill=GOLD)
        for i in range(12):
            a = math.radians(i * 30 - 90); d.line([(gx + math.cos(a) * (r - 40), gy + math.sin(a) * (r - 40)), (gx + math.cos(a) * (r - 12), gy + math.sin(a) * (r - 12))], fill=CREAM, width=8)
        a = math.radians(-90 + 360 * k); d.line([(gx, gy), (gx + math.cos(a) * (r - 50), gy + math.sin(a) * (r - 50))], fill=RED, width=14)
        d.ellipse([gx - 22, gy - 22, gx + 22, gy + 22], fill=RED)
        text_c(d, (gx, gy + 120), '0:%02d' % int(60 * k), font('anton', 90), CREAM, stroke=5)
    elif kind == 'subscribe_odd':
        sparkles(d, t, 24)
        icons = (sc or {}).get('icons', ['emu', 'wave', 'dancer', 'volcano', 'palm', 'pyramid'])
        for i, ic in enumerate(icons):                              # the channel's stories orbit the button
            a = t * 0.7 + i * 2 * math.pi / len(icons)
            depth = 0.5 + 0.5 * math.sin(a)
            mini_icon(d, ic, cx + math.cos(a) * 380, VY0 + 330 + math.sin(a) * 210, 0.7 + 0.5 * depth, t + i)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 620 * p, 160 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 330 - bh / 2, cx + bw / 2, VY0 + 330 + bh / 2], int(bh / 2), fill=(230, 33, 39))
        text_c(d, (cx, VY0 + 330), 'SUBSCRIBE', font('anton', 100 * p), (255, 255, 255))
        sw = math.sin(t * 9) * 0.35 * (1 if (t % 2.0) < 0.8 else 0)        # the bell rings now and then
        bx, by = cx + 300, VY0 + 120
        pts = [(bx + math.sin(sw) * (yy) + math.cos(sw) * xx, by + math.cos(sw) * yy - math.sin(sw) * xx) for xx, yy in ((-60, 90), (-50, 30), (-20, 0), (20, 0), (50, 30), (60, 90))]
        d.polygon(pts, fill=GOLD); d.ellipse([bx - 16 + math.sin(sw) * 110, by + 92, bx + 16 + math.sin(sw) * 110, by + 124], fill=GOLD)
    elif kind == 'liner_night':
        sparkles(d, t, 30); moon(d, 930, VY0 + 40, 50)
        wl = ground - 150
        liner(d, cx + math.sin(t * 0.4) * 24, wl, 1.2, tilt=math.sin(t * 1.1) * 0.008)
        water(d, wl, t)
        for i in range(6):                                                           # reflections of the lights
            rx = cx - 280 + i * 110 + math.sin(t * 2 + i) * 10
            d.line([(rx, wl + 26), (rx + 40, wl + 26)], fill=LIGHT, width=4)
    elif kind == 'portrait':
        sparkles(d, t, 24); py = VY0 + 340
        d.ellipse([cx - 250, py - 320, cx + 250, py + 320], fill=GOLD); d.ellipse([cx - 228, py - 298, cx + 228, py + 298], fill=mix(BG1, CREAM, 0.18))
        d.pieslice([cx - 200, py + 100, cx + 200, py + 420], 180, 360, fill=INK)
        d.rectangle([cx - 38, py + 10, cx + 38, py + 140], fill=INK)
        d.ellipse([cx - 88, py - 170, cx + 88, py + 46], fill=INK); d.ellipse([cx + 34, py - 200, cx + 124, py - 110], fill=INK)
        d.rounded_rectangle([cx - 104, py - 186, cx + 70, py - 150], 16, fill=CREAM)                     # a stewardess's cap
        d.rectangle([cx - 228, py + 262, cx + 228, py + 300], fill=mix(BG1, CREAM, 0.18))
    elif kind == 'collision':
        sparkles(d, t, 20); wl = ground - 150
        k = ease(t / 1.6); hit = t > 1.55; sh = math.sin(t * 50) * 6 * hit * max(0, 1 - (t - 1.55) / 1.5)
        liner(d, cx + 160 + sh, wl, 0.9, hull=mix(INK, STEEL, 0.25))
        warship(d, -260 + 420 * k, wl, 0.62)
        if hit:
            p = ease((t - 1.55) / 0.3); bx, by = 196, wl - 50
            pts = [(bx + math.cos(i * math.pi / 6) * (90 if i % 2 else 34) * p, by + math.sin(i * math.pi / 6) * (90 if i % 2 else 34) * p) for i in range(12)]
            if p < 1 or (t % 0.4) < 0.3: d.polygon(pts, fill=LIGHT)
        water(d, wl, t)
    elif kind == 'iceberg_sink':
        sparkles(d, t, 30); wl = ground - 150
        k = ease((t - 0.8) / max(dur * 0.85 - 0.8, 1))
        iceberg(d, 900, wl, 1.05)
        liner(d, cx - 110, wl + 150 * k, 1.05, tilt=0.3 * k, lights=(t % 0.5) > 0.1 or k < 0.5)
        water(d, wl, t)
        d.polygon([(900 - 170, wl), (900 + 200, wl), (900 + 120, wl + 120), (900 - 120, wl + 120)], fill=mix(LIQ, (226, 242, 255), 0.35))
    elif kind == 'lifeboat':
        sparkles(d, t, 34); moon(d, 200, VY0 + 120, 46); wl = ground - 120
        liner(d, 880, wl + 40, 0.32, tilt=0.6, lights=False)
        water(d, wl, t, dark=True)
        lifeboat(d, cx - 60, wl + 40, 1.25, t)
    elif kind == 'hospital_ship':
        d.ellipse([W - 230, VY0 - 40, W - 90, VY0 + 100], fill=mix(LIGHT, CREAM, 0.4))
        wl = ground - 150
        liner(d, cx + math.sin(t * 0.4) * 20, wl, 1.2, hull=(232, 240, 248), band=(40, 168, 100), cross=True, lights=False, deck=(250, 252, 255))
        water(d, wl, t)
    elif kind == 'propeller':
        d.rectangle([0, VY0 + 40, W, VY1 + 60], fill=shade(LIQ, 0.55))
        for i in range(5):                                                           # light shafts from the surface
            x0 = 120 + i * 220 + math.sin(t + i) * 30
            d.polygon([(x0, VY0 + 40), (x0 + 70, VY0 + 40), (x0 + 160, VY1), (x0 + 60, VY1)], fill=mix(shade(LIQ, 0.55), CREAM, 0.06))
        d.line([(0, VY0 + 40), (W, VY0 + 40)], fill=mix(LIQ, CREAM, 0.5), width=6)
        bx = cx + 40 + ease(t / 2.5) * 140
        d.polygon([(bx - 150, VY0 + 6), (bx + 150, VY0 + 6), (bx + 110, VY0 + 60), (bx - 110, VY0 + 60)], fill=(150, 104, 62))
        d.rectangle([W - 260, VY0 - 60, W + 40, VY0 + 160], fill=shade(STEEL, 0.6))
        propeller(d, W - 150, VY0 + 400, 240, t * 7)
        bubbles(d, t, W - 420, W - 20, VY0 + 80, VY1)
        f = ease((t - 0.8) / 1.4)
        if f > 0:
            px_, py_ = bx - 220, VY0 + 20 + f * 520
            person(d, px_, py_, 2.0, CREAM)
            if f < 0.25: d.ellipse([px_ - 70, VY0 + 26, px_ + 70, VY0 + 56], fill=mix(LIQ, CREAM, 0.7))
        bubbles(d, t, bx - 300, bx - 140, VY0 + 300, VY1, n=10, seed=9)
    elif kind == 'three_ships':
        sparkles(d, t, 20)
        names = ['OLYMPIC', 'TITANIC', 'BRITANNIC']; wl = VY0 + 360
        for i, nm in enumerate(names):
            x = 190 + i * 350; brit = i == 2
            liner(d, x, wl, 0.4, hull=(232, 240, 248) if brit else INK, band=(40, 168, 100) if brit else None, cross=brit, lights=not brit, deck=(250, 252, 255) if brit else None)
            text_c(d, (x, wl + 70), nm, font('anton', 54), CREAM, stroke=4)
            p = back((t - 0.5 - i * 0.6) / 0.4)
            if p > 0.02:
                d.ellipse([x - 54 * p, wl - 300 - 54 * p, x + 54 * p, wl - 300 + 54 * p], fill=(52, 200, 120))
                d.line([(x - 24 * p, wl - 300), (x - 6 * p, wl - 280 * p - 300 * (1 - p)), (x + 28 * p, wl - 324 * p - 300 * (1 - p))], fill=CREAM, width=12)
        water(d, wl + 130, t)
    elif kind == 'follow_sea':
        sparkles(d, t, 26)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 120 - bh / 2, cx + bw / 2, VY0 + 120 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 120), 'FOLLOW', font('anton', 96 * p), CREAM)
        wl = ground - 130
        liner(d, (t * 160) % (W + 900) - 450, wl, 0.6)
        water(d, wl, t)
    elif kind == 'calendar_flip':
        sparkles(d, t, 20)
        seq = (sc or {}).get('seq', [14, 15, 16, 17]); step = 0.55
        i = min(int(t / step), len(seq) - 1); frac = (t - i * step) / step
        cur = seq[i]; nxt = seq[i + 1] if i + 1 < len(seq) else None
        tear_calendar(d, cx, VY0 + 340, 0.88, (sc or {}).get('month', 'NOVEMBER'), cur, sub='', tear=frac if nxt else 0, nxt=nxt)
    elif kind == 'crown':
        sparkles(d, t, 26)
        crown(d, cx, VY0 + 350 + math.sin(t * 1.5) * 10, 1.25, t)
    elif kind == 'canal':
        d.rectangle([60, VY0 + 40, W - 60, ground], fill=(206, 168, 100))
        d.rectangle([60, VY0 + 40, W - 60, VY0 + 150], fill=LIQ); d.rectangle([60, ground - 110, W - 60, ground], fill=LIQ)
        path = [(cx + math.sin(u * 3.0) * 60, VY0 + 150 + u * (ground - 260 - VY0)) for u in [j / 20 for j in range(21)]]
        d.line(path, fill=LIQ, width=54, joint='curve')
        text_c(d, (cx, VY0 + 95), 'MEDITERRANEAN SEA', font('inter', 34), CREAM)
        text_c(d, (cx, ground - 55), 'RED SEA', font('inter', 34), CREAM)
        u = (t * 0.22) % 1; k = min(int(u * 20), 19); f = u * 20 - k
        sx, sy = path[k][0] + (path[k + 1][0] - path[k][0]) * f, path[k][1] + (path[k + 1][1] - path[k][1]) * f
        d.polygon([(sx - 14, sy - 40), (sx, sy - 62), (sx + 14, sy - 40), (sx + 14, sy + 40), (sx - 14, sy + 40)], fill=CREAM)
        d.rectangle([sx - 8, sy - 20, sx + 8, sy + 10], fill=RED)
        p = ease((t - 0.8) / 0.6)
        if p > 0:
            text_c(d, (W - 240, VY0 + 320), '~160 KM', font('anton', 70 * p), INK)
            d.line([(W - 120, VY0 + 170), (W - 120, ground - 130)], fill=INK, width=5)
    elif kind == 'mouse':
        sparkles(d, t, 14)
        monitor(d, cx + 170, VY0 + 250, 0.95, t)
        wooden_mouse(d, cx - 160, ground - 30, 1.15, t)
        d.line([(0, ground - 30), (W, ground - 30)], fill=LINE, width=4)
    elif kind == 'rover':
        sparkles(d, t, 30)
        ex, ey = 870, VY0 + 110
        d.ellipse([ex - 70, ey - 70, ex + 70, ey + 70], fill=(44, 110, 190))
        for bx, by, bw, bh in ((-20, -20, 50, 40), (25, 18, 40, 46), (-30, 30, 30, 22)): d.ellipse([ex + bx - bw / 2, ey + by - bh / 2, ex + bx + bw / 2, ey + by + bh / 2], fill=(110, 180, 110))
        d.polygon([(x_, ground - 150 + 850 * (1 - math.sqrt(max(0, 1 - ((x_ - cx) / (cx + 260)) ** 2)))) for x_ in range(-260, W + 261, 30)] + [(W + 260, ground + 40), (-260, ground + 40)], fill=(170, 166, 160))
        for cx_, cy_, r in ((160, ground - 40, 60), (700, ground - 90, 40), (930, ground - 20, 80)): d.ellipse([cx_ - r, cy_ - r * 0.36, cx_ + r, cy_ + r * 0.36], fill=(130, 126, 122))
        rx = 120 + (t * 70) % 900
        for j in range(10):                                                         # wheel tracks behind it
            tx = rx - 180 - j * 46
            if tx > -40: d.rectangle([tx, ground - 66, tx + 26, ground - 58], fill=(140, 136, 130))
        lunar_rover(d, rx, ground - 56, 1.05, t)
    elif kind == 'protest':
        buildings(d, 60, ground - 60, n=3, seed=4, scale=0.9); buildings(d, 620, ground - 60, n=3, seed=7, scale=0.9)
        for tx in (480, 600):                                                       # Prague's twin spires
            d.polygon([(tx - 40, ground - 60), (tx - 40, ground - 420), (tx, ground - 560), (tx + 40, ground - 420), (tx + 40, ground - 60)], fill=shade(STEEL, 0.4))
        rng = random.Random(6)
        for r in range(3):
            for c in range(9):
                x = 70 + c * 118 + (r % 2) * 50; y = ground - 30 - (2 - r) * 70; b = abs(math.sin(t * 3 + c + r)) * 8
                col = mix(BG1, CREAM, 0.3 + 0.25 * r)
                person(d, x, y - b, 1.1 + 0.15 * r, col)
                if (c + r) % 3 == 0: candle(d, x + 30, y - 30 - b, 1.0 + 0.1 * r, t, c)
                elif (c + r) % 3 == 1 and math.sin(t * 6 + c) > 0:                  # jingling keys
                    kx, ky = x + 30, y - 110 - b; d.ellipse([kx - 12, ky - 12, kx + 12, ky + 12], outline=GOLD, width=5); d.line([(kx, ky + 10), (kx, ky + 46)], fill=GOLD, width=6)
    elif kind == 'follow_calendar':
        sparkles(d, t, 20)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 80 - bh / 2, cx + bw / 2, VY0 + 80 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 80), 'FOLLOW', font('anton', 96 * p), CREAM)
        tear_calendar(d, cx, VY0 + 470, 0.55, (sc or {}).get('month', 'NOVEMBER'), (sc or {}).get('day', 17))
    elif kind == 'moon_earth':
        sparkles(d, t, 40)
        earth(d, 230, ground - 120, 140); big_moon(d, 830, VY0 + 170, 150)
        pts = [(230 + (830 - 230) * u, ground - 120 - math.sin(u * math.pi) * 380 - (ground - 120 - VY0 - 170) * u) for u in [j / 40 for j in range(41)]]
        for j in range(0, 40, 2): d.line([pts[j], pts[j + 1]], fill=CREAM, width=5)
        u = (t * 0.18) % 1; k = min(int(u * 40), 39); sx, sy = pts[k]
        d.ellipse([sx - 14, sy - 14, sx + 14, sy + 14], fill=GOLD)
    elif kind == 'launch':
        sparkles(d, t, 26)
        k = ease((t - 0.8) / max(dur - 0.8, 1.5)) ** 1.6; base = ground - 40 - k * 160
        d.rectangle([150, ground - 600, 190, ground - 40], fill=shade(RED, 0.7))
        for j in range(12): d.line([(150, ground - 40 - j * 48), (190, ground - 64 - j * 48)], fill=shade(RED, 0.5), width=4)
        d.rectangle([60, ground - 40, W - 60, ground], fill=shade(STEEL, 0.5))
        saturn_v(d, cx + 40, base, 0.7, flame=0.45 * min(1, t / 0.6), t=t)
        puff_cloud(d, cx + 40, ground - 30, t, n=18, spread=460)
    elif kind == 'orbit':
        sparkles(d, t, 40)
        big_moon(d, cx, VY0 + 360, 230)
        a = t * 1.1; rx_, ry_ = 380, 120
        x_, y_ = cx + math.cos(a) * rx_, VY0 + 360 + math.sin(a) * ry_
        if math.sin(a) > -0.2 or abs(math.cos(a)) > 0.62:
            d.rectangle([x_ - 30, y_ - 14, x_ + 18, y_ + 14], fill=(220, 224, 232)); d.polygon([(x_ + 18, y_ - 14), (x_ + 46, y_), (x_ + 18, y_ + 14)], fill=(200, 204, 214))
        d.arc([cx - rx_, VY0 + 360 - ry_, cx + rx_, VY0 + 360 + ry_], 0, 180, fill=mix(BG1, CREAM, 0.35), width=3)
    elif kind == 'landing':
        sparkles(d, t, 30); earth(d, 880, VY0 + 90, 60)
        moon_ground(d, ground)
        k = ease(t / max(dur * 0.8, 1.5)); y_ = ground - 110 - 470 * (1 - k)
        if k < 0.98:
            for i in range(10):
                a = math.radians(180 + i * 18); r = 80 + 260 * (1 - (ground - 110 - y_) / 470) * ((t * 3 + i) % 1)
                d.ellipse([cx + math.cos(a) * r - 20, ground - 80 - 12, cx + math.cos(a) * r + 20, ground - 80 + 12], fill=shade(MOONC, 1.1))
            d.polygon([(cx - 22, y_ + 20), (cx, y_ + 90 + 20 * math.sin(t * 30)), (cx + 22, y_ + 20)], fill=(255, 170, 60))
        descent_stage(d, cx, y_ + 30, 1.0); ascent_stage(d, cx, y_ - 110, 1.0)
        if (t % 0.8) < 0.45 and k < 0.9:
            d.rounded_rectangle([60, VY0 + 30, 340, VY0 + 120], 16, fill=RED); text_c(d, (200, VY0 + 75), 'ALARM 1202', font('anton', 52), CREAM)
        fuel = max(0.04, 1 - k * 0.96); d.rounded_rectangle([W - 120, VY0 + 160, W - 70, VY0 + 460], 12, outline=CREAM, width=5)
        d.rectangle([W - 112, VY0 + 452 - 284 * fuel, W - 78, VY0 + 452], fill=GOLD if fuel > 0.2 else RED)
        text_c(d, (W - 95, VY0 + 500), 'FUEL', font('inter', 30), CREAM)
    elif kind == 'footprint':
        sparkles(d, t, 30)
        d.rectangle([60, VY0 + 40, W - 60, ground], fill=MOONC)
        rng = random.Random(2)
        for i in range(60):
            x_, y_ = rng.uniform(80, W - 80), rng.uniform(VY0 + 60, ground - 20); r = rng.uniform(4, 14)
            d.ellipse([x_ - r, y_ - r, x_ + r, y_ + r], fill=shade(MOONC, rng.uniform(0.82, 1.08)))
        p = ease(t / 0.8)
        boot_print(d, cx - 120, VY0 + 400, 1.15 * p + 0.01, -0.12)
        if t > 0.9: boot_print(d, cx + 160, VY0 + 260, 1.15 * ease((t - 0.9) / 0.6) + 0.01, 0.12)
    elif kind == 'moon_liftoff':
        sparkles(d, t, 30); earth(d, 200, VY0 + 110, 70)
        moon_ground(d, ground)
        descent_stage(d, cx, ground - 30, 1.15)
        k = ease((t - 0.6) / max(dur - 0.6, 1.5)) ** 1.4; y_ = ground - 30 - 150 * 1.15 - k * 220
        if t > 0.4: d.polygon([(cx - 26, y_), (cx, y_ + 120 + 20 * math.sin(t * 30)), (cx + 26, y_)], fill=(255, 170, 60))
        ascent_stage(d, cx, y_, 1.15)
    elif kind == 'splashdown':
        sparkles(d, t, 20); wl = ground - 160
        k = ease(t / max(dur * 0.75, 1.5)); y_ = wl - 20 - 150 * (1 - k)
        chutes(d, cx, y_, 0.8, t); capsule(d, cx, y_, 1.3)
        water(d, wl, t)
        if k > 0.97:
            r = 60 + ((t * 120) % 200)
            d.ellipse([cx - r * 1.6, wl - r * 0.2, cx + r * 1.6, wl + r * 0.2], outline=mix(LIQ, CREAM, 0.6), width=5)
    elif kind == 'follow_space':
        sparkles(d, t, 40)
        p = 1 + 0.06 * math.sin(t * 6); bw, bh = 560 * p, 150 * p
        d.rounded_rectangle([cx - bw / 2, VY0 + 100 - bh / 2, cx + bw / 2, VY0 + 100 + bh / 2], int(bh / 2), fill=RED)
        text_c(d, (cx, VY0 + 100), 'FOLLOW', font('anton', 96 * p), CREAM)
        big_moon(d, cx, VY0 + 470, 170)
        a = t * 0.9; d.ellipse([cx + math.cos(a) * 280 - 12, VY0 + 470 + math.sin(a) * 90 - 12, cx + math.cos(a) * 280 + 12, VY0 + 470 + math.sin(a) * 90 + 12], fill=GOLD)
    elif kind == 'war_clock':
        sparkles(d, t, 20)
        k = min(1, t / max(dur * 0.85, 1.5)); m_ = 38 * ease(k)
        clock_face(d, cx, VY0 + 330, 250, 9, 2 + m_)
        d.pieslice([cx - 230, VY0 + 100, cx + 230, VY0 + 560], -90 + 12, -90 + 12 + 6 * m_, fill=mix(CREAM, RED, 0.35))
        clock_face_hands = None
        ha = math.radians((9 + (2 + m_) / 60) * 30 - 90); ma = math.radians((2 + m_) * 6 - 90)
        d.line([(cx, VY0 + 330), (cx + math.cos(ha) * 130, VY0 + 330 + math.sin(ha) * 130)], fill=INK, width=16)
        d.line([(cx, VY0 + 330), (cx + math.cos(ma) * 200, VY0 + 330 + math.sin(ma) * 200)], fill=RED, width=10)
        d.ellipse([cx - 16, VY0 + 314, cx + 16, VY0 + 346], fill=INK)
        text_c(d, (cx, ground - 40), '9:%02d AM' % int(2 + m_), font('anton', 84), CREAM, stroke=6)
    elif kind == 'island_palace':
        sun_disc(d, 860, VY0 + 130, 90, col=(255, 226, 150)); wl = ground - 130
        d.polygon([(90, wl), (180, wl - 120), (900, wl - 130), (1000, wl)], fill=(222, 196, 140))
        palace(d, cx + 40, wl - 120, 0.62, t=t); palm(d, 170, wl - 110, 0.55, 0.3, t, col=(40, 90, 60)); palm(d, 930, wl - 110, 0.5, -0.3, t, col=(40, 90, 60))
        water(d, wl, t)
        p = back((t - 0.8) / 0.4)
        if p > 0.02:
            d.rounded_rectangle([60, VY0 + 40, 60 + 330 * p, VY0 + 120], 20, fill=INK)
            if p > 0.8: text_c(d, (225, VY0 + 80), 'EAST AFRICA', font('anton', 52), CREAM)
    elif kind == 'seize':
        wl = ground - 60; d.rectangle([60, wl, W - 60, ground + 20], fill=(222, 196, 140))
        palace(d, cx, wl, 1.0, flag=ease((t - 0.6) / 1.4), t=t)
        for i, x in enumerate((150, 250, 830, 930)):
            person(d, x, wl - 34, 1.6, INK); d.line([(x + 26, wl - 60), (x + 34, wl - 210)], fill=INK, width=8)
    elif kind == 'ultimatum':
        wl = ground - 120
        clock_face(d, cx, VY0 + 250, 190, 8, 40 + 20 * ease(t / max(dur * 0.8, 1.5)))
        for i, x in enumerate((-80, 330, 700)):
            warship(d, x + math.sin(t * 0.7 + i) * 12 + 120, wl, 0.48 + 0.06 * (i % 2))
        water(d, wl, t)
    elif kind == 'bombard':
        wl = ground - 110
        palace(d, W - 250, wl, 0.62, damage=ease(t / max(dur * 0.9, 1.5)) * 0.6, t=t)
        ships = [(130, wl - 60), (380, wl - 50)]
        for x, y in ships: warship(d, x - 30, wl, 0.42)
        shell_arcs(d, t, [(x + 100, y - 20) for x, y in ships], W - 250, wl - 140)
        puff_cloud(d, W - 250, wl - 150, t, n=10, spread=220, col=(120, 116, 112))
        water(d, wl, t)
    elif kind == 'sink_small':
        wl = ground - 130; k = ease((t - 0.6) / max(dur * 0.85, 1.5))
        liner(d, cx, wl + 170 * k, 0.75, tilt=-0.35 * k, hull=(232, 240, 248), funnels=1, lights=False, deck=(250, 252, 255))
        puff_cloud(d, cx + 80, wl - 200 + 120 * k, t, n=10, spread=160, col=(110, 106, 104))
        water(d, wl, t)
    elif kind == 'ruins':
        wl = ground - 60; d.rectangle([60, wl, W - 60, ground + 20], fill=(222, 196, 140))
        palace(d, cx, wl, 1.0, damage=1.0, flag=max(0.0, 1 - ease(t / 1.6)), t=t)
        puff_cloud(d, cx - 100, wl - 260, t, n=14, spread=300, col=(96, 92, 90))
        puff_cloud(d, cx + 160, wl - 200, t + 1.3, n=10, spread=200, col=(110, 104, 100))
    elif kind == 'consulate':
        sparkles(d, t, 18); wl = ground - 40
        d.rectangle([cx - 300, wl - 420, cx + 300, wl], fill=(214, 206, 190)); d.polygon([(cx - 330, wl - 420), (cx, wl - 540), (cx + 330, wl - 420)], fill=(170, 160, 146))
        for i in range(4): d.rectangle([cx - 250 + i * 140, wl - 360, cx - 190 + i * 140, wl - 260], fill=shade(WALL, 0.55))
        d.rectangle([cx - 50, wl - 170, cx + 50, wl], fill=(110, 76, 46))
        for i, c in enumerate(((0, 0, 0), (214, 30, 30), (240, 240, 240))):             # the German imperial flag of the time
            d.rectangle([cx + 300, wl - 600 + i * 30, cx + 440, wl - 570 + i * 30], fill=c)
        d.line([(cx + 300, wl - 600), (cx + 300, wl - 420)], fill=INK, width=6)
        x = W - 140 - min(1, t / 2.2) * 520
        person(d, x, wl - 34, 1.8, INK)
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

    bases = base_frames(); pal_key = json.dumps(scene_palette(spec, {}), sort_keys=True)
    wipe = spec.get('transitions', False)
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
        key = json.dumps(scene_palette(spec, sc), sort_keys=True)
        if key != pal_key:                                   # this scene has its own colours
            set_palette(scene_palette(spec, sc)); bases = base_frames(3); pal_key = key
        img = compose(bases[fi % len(bases)], sc, lt, t / total)
        if wipe and sc is not scenes[0] and lt < 0.28:        # a colour bar sweeps off to reveal the scene
            k = ease(lt / 0.28); dr = ImageDraw.Draw(img); x0 = FW * S * k
            dr.rectangle([x0, 0, FW * S, FH * S], fill=GOLD); dr.rectangle([x0, 0, x0 + 18 * S, FH * S], fill=RED)
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
