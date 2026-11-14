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
    rng = np.random.default_rng(1); g = 2 * S
    out = []
    for _ in range(k):
        n = rng.normal(0, 7, (RH // g, RW // g, 1)).astype(np.float32).repeat(g, 0).repeat(g, 1)
        out.append(Image.fromarray(np.clip(img + n, 0, 255).astype(np.uint8)).convert('RGBA'))
    return out

def text_c(d, xy, s, f, fill, anchor='mm', stroke=0, sfill=INK):
    d.text(xy, s, font=f, fill=fill, anchor=anchor, stroke_width=stroke, stroke_fill=sfill)

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

# ---------- visuals (the area between y=640 and y=1330) ----------
VY0, VY1 = 640, 1330
def visual(d, kind, t, dur, img):
    cx, ground = W / 2, VY1 - 40
    d.line([(60, ground), (W - 60, ground)], fill=(90, 72, 52), width=4)
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
        visual(d, sc['visual'], lt, sc['dur'], img)
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
    if '--scale' in args:                       # --scale 1 = 1080p, 2 = 4K (default)
        i = args.index('--scale'); S = float(args[i + 1]); del args[i:i + 2]
    main(args[0], args[1])
