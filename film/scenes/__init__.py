"""Scene registry: the whole film is the list returned by all_scenes()."""
from .. import story as S
from .base import Captions, ChapterTag


def all_scenes():
    from .opening import ColdOpen, Title
    from .waves import Circles, Euler3D, TimeFreq3D
    body = [ColdOpen(), Title(), Euler3D(), Circles(), TimeFreq3D()]
    try:
        from .drawing import drawing_scenes
        body += drawing_scenes()
    except ImportError:
        pass
    try:
        from .sound import sound_scenes
        body += sound_scenes()
    except ImportError:
        pass
    try:
        from .image import image_scenes
        body += image_scenes()
    except ImportError:
        pass
    try:
        from .beyond import beyond_scenes
        body += beyond_scenes()
    except ImportError:
        pass
    try:
        from .family import family_scenes
        body += family_scenes()
    except ImportError:
        pass
    from .finale import Finale, Heat3D, Montage
    scenes = body + [Heat3D(), Montage(body), Finale()]
    _configure(scenes)
    return scenes


# hand-overs between shots: (kind, seconds before the cut, seconds after, question card for chapter irises)
TRANS = {
    S.T_EULER: ("iris", 0.6, 0.95, "从一个圆说起"),
    S.T_SQUARE: ("zoom", 0.2, 0.25, None),
    S.T_PANELS: ("iris", 0.6, 0.95, "那么，任何图形呢？"),
    S.T_BUTTERFLY: ("zoom", 0.2, 0.25, None),
    S.T_QUICK: ("push_l", 0.2, 0.2, None),
    S.T_PTOLEMY: ("zoom", 0.2, 0.25, None),
    S.T_CHORD: ("iris", 0.6, 0.95, "声音呢？"),
    S.T_WIND: ("push_l", 0.2, 0.2, None),
    S.T_EAR: ("zoom", 0.2, 0.25, None),
    S.T_SHAZAM: ("push_r", 0.2, 0.2, None),
    S.T_DROP: ("zoom", 0.15, 0.2, None),
    S.T_NOISE: ("push_l", 0.2, 0.2, None),
    S.T_WAVES2D: ("iris", 0.6, 0.95, "图像呢？"),
    S.T_IMGBUILD: ("zoom", 0.2, 0.25, None),
    S.T_SPEC3D: ("push_l", 0.2, 0.2, None),
    S.T_FILTER: ("push_r", 0.2, 0.2, None),
    S.T_JPEG: ("zoom", 0.2, 0.25, None),
    S.T_BEYOND: ("iris", 0.6, 0.95, "只是声音和图像吗？"),
    S.T_LAPLACE: ("iris", 0.6, 0.95, "只有傅里叶变换吗？"),
    S.T_WAVELET: ("push_l", 0.2, 0.2, None),
    S.T_HEAT: ("iris", 0.6, 0.95, "这一切，从哪里开始？"),
}
for _i, (_n, _t) in enumerate(S.VIGNETTES):
    TRANS[_t] = (["zoom", "push_l", "push_r"][_i % 3], 0.2, 0.2, None)

# sub-frames of motion blur for the fast-moving shots
BLUR = {"ColdOpen": 2, "Title": 3, "Euler3D": 3, "Circles": 3, "TimeFreq3D": 2, "Panels": 2, "Butterfly3D": 3,
        "QuickDraws": 4, "Ptolemy": 4, "Winding": 3, "Drop3D": 2, "GW": 4, "FFT": 2, "Heat3D": 2, "Finale": 2,
        "SphHarm": 2, "Quantum": 2, "Tides": 2, "WiFi": 2}


def _configure(scenes):
    for sc in scenes:
        for t0, tr in TRANS.items():
            if abs(sc.start - t0) < 1e-6 and type(sc).__name__ not in ("Montage",):
                sc.trans = tr
        sc.blur = BLUR.get(type(sc).__name__, 1)


CAPTIONS = [
    # cold open
    (2.3, 5.7, "1807 年，傅里叶断言：", "In 1807, Fourier made a claim:"),
    (5.9, 8.8, "任何曲线，都是正弦波之和", "any curve is a sum of sine waves"),
    # 01 circles & waves
    (19.0, 21.8, "旋转的点：正面看是圆，侧面看是正弦波", "A spinning point: a circle head-on, a sine wave side-on"),
    (22.3, 25.6, "方波 = 奇次谐波之和", "A square wave is a sum of odd harmonics"),
    (25.8, 29.9, "每个圆都是一个泛音——听，音色在变亮", "Each circle is an overtone. Hear the tone brighten"),
    (30.3, 33.0, "转过 90°，波形变成频谱", "Turn 90 degrees: the waveform becomes a spectrum"),
    (33.2, 35.8, "这就是傅里叶变换", "That is the Fourier transform"),
    (36.2, 37.9, "锯齿波：全部谐波", "Sawtooth: every harmonic"),
    (38.2, 41.8, "心电监护仪先滤掉 50 Hz 工频干扰", "ECG monitors first filter out 50 Hz mains hum"),
    # 02 drawing
    (43.0, 47.6, "半径是振幅，起始角是相位", "Radius is amplitude, starting angle is phase"),
    (48.4, 51.8, "沿时间轴展开，笔尖画出一件三维雕塑", "Unrolled along time, the pen draws a 3D sculpture"),
    (55.0, 57.8, "压回平面：600 个圆，一只蝴蝶", "Flattened: 600 circles, one butterfly"),
    (58.3, 61.8, "换一组系数，就能画出任何东西", "New coefficients draw anything"),
    (62.3, 65.8, "托勒密的“本轮”：圆套圆，解释火星逆行", "Ptolemy's epicycles explained the loops of Mars"),
    # 03 sound
    (67.0, 71.6, "声音是空气的振动，和弦由三个正弦波组成", "Sound is vibrating air; this chord is three sine waves"),
    (72.3, 75.8, "把声波缠绕在圆上，改变转速", "Wrap the sound around a circle and change the speed"),
    (76.0, 79.8, "对上音高时重心跳出——这就是傅里叶积分", "At each note the centre of mass jumps: the Fourier integral"),
    (80.3, 83.8, "耳蜗：约 3500 个毛细胞，按频率排开", "The cochlea: ~3,500 hair cells sorted by pitch"),
    (84.3, 87.8, "听歌识曲：频谱亮点连成指纹", "Song ID: spectrogram peaks become a fingerprint"),
    (88.4, 92.8, "你正在听的配乐，它的三维频谱", "The music you are hearing, as a 3D spectrum"),
    (94.7, 97.2, "降噪耳机：反相波抵消噪声", "Noise cancelling: an inverted wave cancels the noise"),
    (97.4, 99.8, "MP3：删掉听不见的频率", "MP3 drops the frequencies you cannot hear"),
    # 04 images
    (101.0, 103.8, "图像的“正弦波”，是起伏的波纹", "An image's sine waves are ripples"),
    (106.5, 111.8, "从最强的波纹叠起：只用 1%，猫已清晰可辨", "Strongest ripples first: at 1% the cat is clear"),
    (112.5, 117.6, "照片的能量集中在低频，所以能被大幅压缩", "A photo's energy sits in the lows, so it compresses well"),
    (118.3, 120.8, "去掉高频：画面变模糊", "Remove the highs: it blurs"),
    (121.0, 123.8, "去掉低频：只剩轮廓", "Remove the lows: only edges remain"),
    (124.3, 129.8, "JPEG：只保留低频波纹，文件小十倍", "JPEG keeps the low-frequency patterns: 10x smaller"),
    # beyond (4 s each)
    (134.3, 137.8, "氦，先在太阳光谱中被发现", "Helium was found in the Sun's spectrum first"),
    (138.3, 141.8, "衍射图，就是孔径的傅里叶变换", "A diffraction pattern is the aperture's Fourier transform"),
    (142.3, 145.8, "衍射图里的“X”，暴露了双螺旋", "The X in the pattern revealed the double helix"),
    (146.3, 149.8, "MRI 测的是频率，逆变换还原出大脑", "MRI measures frequencies; the inverse transform shows the brain"),
    (150.3, 153.8, "真实数据：两个黑洞合并的“啁啾”", "Real data: the chirp of two merging black holes"),
    (154.3, 157.8, "球面上的傅里叶：原子轨道、天气预报", "Fourier on a sphere: atomic orbitals, weather models"),
    (158.3, 161.8, "潮汐：几十个天文周期的叠加", "Tides: dozens of astronomical cycles added up"),
    (162.3, 165.8, "上千个子载波并行，互不干扰", "Thousands of subcarriers side by side"),
    (166.3, 169.8, "位置越确定，动量越模糊", "The sharper the position, the blurrier the momentum"),
    (170.3, 173.8, "大模型用正弦波标记词的位置", "Language models tag word positions with sine waves"),
    (174.3, 177.8, "FFT：N² 变成 N log N，快了五万倍", "FFT: N² becomes N log N, 50,000x faster"),
    # the family
    (179.0, 181.3, "拉普拉斯变换：把频率推广到复平面", "Laplace: frequency extended to the complex plane"),
    (181.5, 183.8, "沿虚轴切开，就是傅里叶变换", "Slice the imaginary axis: the Fourier transform"),
    (184.3, 186.9, "短时傅里叶：窗口固定", "Short-time Fourier: one fixed window"),
    (187.1, 189.8, "小波：高频看清时刻，低频看清音高", "Wavelets: timing for highs, pitch for lows"),
    # heat & montage
    (191.0, 193.8, "一切的起点：热方程", "Where it began: the heat equation"),
    (194.0, 197.8, "高频最先消失——傅里叶由此发明了级数", "The highs fade first; Fourier's series was born here"),
    (198.3, 201.8, "从 1807 年的一根铁棒……", "From an iron bar in 1807..."),
    (202.0, 205.8, "到你口袋里的每一首歌、每一张照片", "to every song and photo in your pocket"),
    # finale: the card flips to show the flower's spectrum
    (211.0, 215.3, "翻过来看：这朵花的频谱，只有 4 根谱线", "Turned sideways, the flower is just four spectral lines"),
]

CHAPTERS = [
    (S.T_EULER + 0.2, S.T_PANELS - 0.2, "01", "圆 与 波", "CIRCLES & WAVES"),
    (S.T_PANELS + 0.2, S.T_CHORD - 0.2, "02", "以 圆 作 画", "DRAWING WITH CIRCLES"),
    (S.T_CHORD + 0.2, S.T_WAVES2D - 0.2, "03", "声 音", "SOUND"),
    (S.T_WAVES2D + 0.2, S.T_BEYOND - 0.2, "04", "图 像", "IMAGES"),
    (S.T_LAPLACE + 0.2, S.T_HEAT - 0.2, "05", "变 换 家 族", "THE FAMILY"),
]


def overlays():
    return [ChapterTag(CHAPTERS), Captions(CAPTIONS)]
