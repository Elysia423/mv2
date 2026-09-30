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
    try:
        from .finale import Finale, Heat3D, Montage
        return body + [Heat3D(), Montage(body), Finale()]
    except ImportError:
        return body


CAPTIONS = [
    # cold open
    (2.4, 5.6, "1807 年，巴黎。傅里叶提交论文《论热的传播》", "Paris, 1807. Fourier submits his memoir on the propagation of heat"),
    (5.9, 8.8, "他断言：任何函数，都能写成正弦波之和", "He claimed: any function can be written as a sum of sine waves"),
    # 01 circles & waves
    (18.4, 21.8, "欧拉公式：旋转的点，正面看是圆，侧面看是正弦波", "Euler's formula: a spinning point is a circle head-on, a sine wave side-on"),
    (22.3, 25.3, "方波 = 奇次谐波之和，振幅依次是 1、1/3、1/5……", "A square wave is a sum of odd harmonics: 1, 1/3, 1/5..."),
    (25.5, 27.9, "每个圆都是一个泛音——听，音色正在变亮", "Each circle is an overtone; hear the tone get brighter"),
    (28.1, 29.9, "吉布斯现象：跳变处永远多冲出约 9%", "Gibbs phenomenon: a ~9% overshoot that never goes away"),
    (30.3, 32.9, "转过 90°：时域里的波形，变成频域里的谱线", "Turn 90 degrees: a waveform in time becomes lines in frequency"),
    (33.2, 35.8, "这就是傅里叶变换：同一个信号，换一套坐标", "That is the Fourier transform: the same signal in new coordinates"),
    (36.3, 37.9, "锯齿波：包含全部谐波，振幅是 1/n", "Sawtooth: every harmonic, amplitude 1/n"),
    (38.3, 41.8, "心电监护仪要先滤掉 50 Hz 的工频干扰，才能看清每一次心跳", "ECG monitors filter out 50 Hz mains hum to see each beat"),
    # 02 drawing
    (42.3, 45.0, "让圆在平面上转：半径 = 振幅，初始角度 = 相位", "Circles in a plane: radius is amplitude, starting angle is phase"),
    (45.2, 47.8, "8、24、80、400 个圆：项数越多，误差越小", "8, 24, 80, 400 circles: more terms, less error"),
    (48.3, 51.4, "600 个圆，转速都是基频的整数倍", "600 circles, each spinning at a whole multiple of one frequency"),
    (51.7, 54.7, "沿时间轴展开，笔尖的轨迹是一件三维雕塑", "Unrolled along time, the pen traces a sculpture in 3D"),
    (55.0, 57.8, "压回平面，就是一只蝴蝶", "Flatten it, and it is a butterfly"),
    (58.2, 61.8, "换一组系数，同一台“圆的机器”能画出任何东西", "Change the coefficients and the same machine draws anything"),
    (62.3, 65.8, "公元 150 年，托勒密就用“本轮”解释火星逆行——圆套圆，古已有之", "c. 150 AD: Ptolemy's epicycles explained the retrograde loops of Mars"),
    # 03 sound
    (66.3, 68.9, "声音是空气压力的波动：这是一个 A 小三和弦", "Sound is a pressure wave: this is an A minor chord"),
    (69.1, 71.8, "拆开它：220 Hz、262 Hz、330 Hz 三个正弦波", "Unmixed: three sine waves at 220, 262 and 330 Hz"),
    (72.3, 75.8, "把波形缠绕在圆上，慢慢改变缠绕的频率……", "Wrap the wave around a circle and sweep the winding frequency..."),
    (76.0, 79.8, "频率对上时，重心猛地偏离圆心——这就是傅里叶积分", "When it matches a note, the centre of mass jumps: the Fourier integral"),
    (80.3, 83.8, "耳蜗：约 3500 个内毛细胞沿螺旋排开，从 20 kHz 到 20 Hz", "The cochlea: ~3,500 inner hair cells tuned from 20 kHz down to 20 Hz"),
    (84.3, 87.8, "听歌识曲：取出频谱里的亮点，连成一张“星图”指纹", "Song recognition turns spectrogram peaks into a constellation fingerprint"),
    (88.4, 91.8, "三维频谱：横向是频率，高度是能量，纵深是时间", "A 3D spectrogram: frequency across, energy up, time receding"),
    (92.0, 94.3, "这就是此刻正在播放的配乐", "This is the soundtrack playing right now"),
    (94.7, 97.3, "降噪耳机：生成反相的波，与噪声相加归零", "Noise cancelling: add the inverted wave and the noise sums to zero"),
    (97.5, 99.8, "MP3：删去被掩蔽、听不见的频率，体积只剩约 1/10", "MP3 drops masked, inaudible frequencies: about 1/10 the size"),
    # 04 images
    (100.3, 103.8, "图像是二维信号，它的“正弦波”是一道道起伏的波纹", "Images are 2D signals; their sine waves are ripples"),
    (104.3, 107.8, "按能量从大到小，把波纹一道道叠加……", "Add the ripples, strongest first..."),
    (108.0, 111.8, "只用 1% 的波纹，猫已经清晰可辨", "With 1% of the ripples, the cat is already clear"),
    (112.3, 115.1, "照片的频谱：能量集中在中心的低频", "A photo's spectrum: most energy sits in the low frequencies"),
    (115.3, 117.8, "高频能量比低频弱几个数量级——所以图像能被大幅压缩", "Highs are orders of magnitude weaker: that is why images compress"),
    (118.3, 120.8, "低通：只留低频——模糊，就像失焦", "Low-pass: keep the lows, and it blurs like defocus"),
    (121.0, 123.8, "高通：只留高频——只剩边缘，像一幅素描", "High-pass: keep the highs, and only the edges remain"),
    (124.3, 127.0, "JPEG：切成 8×8 小块，用 64 种余弦波纹（DCT）表示", "JPEG: 8x8 blocks written in 64 cosine patterns (the DCT)"),
    (127.2, 129.8, "丢掉人眼不敏感的高频，文件缩小约 10 倍", "Drop the highs the eye barely notices: about 10x smaller"),
    # beyond (4 s each)
    (134.3, 137.8, "1868 年，氦先在太阳光谱中被发现，27 年后才在地球上找到", "Helium was found in the Sun's spectrum in 1868, 27 years before on Earth"),
    (138.3, 141.8, "远场衍射图就是孔径的傅里叶变换：韦布望远镜的六道星芒由此而来", "Diffraction is the aperture's Fourier transform: hence Webb's six-pointed stars"),
    (142.3, 145.8, "X 射线衍射图中的“X”，暴露了 DNA 的双螺旋结构", "The X in the diffraction photo gave away DNA's double helix"),
    (146.3, 149.8, "核磁共振直接测量频率数据，逆傅里叶变换还原出大脑", "MRI measures frequency data; an inverse transform rebuilds the brain"),
    (150.3, 153.8, "13 亿光年外黑洞合并：0.2 秒内频率从 35 Hz 升到 250 Hz", "Black holes merge 1.3 billion light-years away: 35 to 250 Hz in 0.2 s"),
    (154.3, 157.8, "原子轨道、宇宙微波背景、全球天气预报，都用它展开", "Atomic orbitals, the cosmic microwave background, global weather models"),
    (158.3, 161.8, "潮汐是几十个天文周期之和，主太阴半日潮 M2 = 12.42 小时", "Tides are dozens of astronomical cycles added up; M2 = 12.42 hours"),
    (162.3, 165.8, "数百个子载波并行传输，每个峰值处其余载波恰好为零", "Hundreds of subcarriers side by side, each zero at the others' peaks"),
    (166.3, 169.8, "位置与动量互为傅里叶变换：一个越窄，另一个就越宽", "Position and momentum are a Fourier pair: squeeze one, the other spreads"),
    (170.3, 173.8, "大模型用不同频率的正弦波，给每个词标上位置", "Language models mark each token's position with sine waves"),
    (174.3, 177.8, "N² → N log N。其实高斯在 1805 年就想到了这个算法", "N² to N log N. Gauss had found the same trick in 1805"),
    # the family
    (178.3, 180.9, "拉普拉斯变换：把频率推广到整个复平面，极点决定系统是否稳定", "Laplace: extend frequency to the complex plane; poles decide stability"),
    (181.1, 183.8, "沿虚轴切一刀，就是傅里叶变换：滤波器与控制器都靠它设计", "Slice along the imaginary axis to get the Fourier transform"),
    (184.3, 186.9, "短时傅里叶：窗口固定，时间与频率的分辨率此消彼长", "Short-time Fourier: one window size, one fixed trade-off"),
    (187.1, 189.8, "小波：高频看清时刻，低频看清音高——JPEG 2000 就用它", "Wavelets: sharp timing for highs, sharp pitch for lows; used in JPEG 2000"),
    # heat & montage
    (190.3, 193.8, "一切的起点：热方程。每个频率按 e^(−n²t) 衰减", "Where it began: the heat equation. Each frequency decays like e^(−n²t)"),
    (194.0, 197.8, "高频最先消失——为了解这个方程，傅里叶发明了他的级数", "The highs die first; to solve it, Fourier invented his series"),
    (198.3, 201.8, "从 1807 年的一根铁棒……", "From an iron bar in 1807..."),
    (202.0, 205.8, "到你口袋里的每一次通话、每一张照片、每一首歌", "to every call, photo and song in your pocket"),
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
