"""Scene registry: the whole film is the list returned by all_scenes()."""
from .base import Captions, ChapterTag


def all_scenes():
    from .opening import ColdOpen, Title
    from .circles import Circles, Drawing
    from .sound import Chord, Drop, Ear
    from .image import Images
    from .beyond import beyond_scenes
    from .finale import Finale, Heat, Montage
    body = [ColdOpen(), Title(), Circles(), Drawing(), Chord(), Ear(), Drop(), Images()] + beyond_scenes()
    return body + [Heat(), Montage(body), Finale()]


CAPTIONS = [
    (3.6, 7.6, "1807 年，巴黎", "Paris, 1807"),
    (8.0, 11.8, "傅里叶宣称：任何曲线，都能由旋转的圆叠加而成", "Fourier claimed: any curve can be built from spinning circles"),
]

CAPTIONS += [
    (24.6, 29.2, "一个旋转的圆，投影出最纯粹的波：正弦波", "One spinning circle traces the purest wave: a sine"),
    (29.8, 33.8, "再加一个更小、转得更快的圆……", "Add a smaller circle that spins faster..."),
    (34.2, 36.9, "圆越多，波形越接近方波", "The more circles, the closer to a square wave"),
    (37.1, 39.9, "听——每一个圆，都是一个泛音", "Listen: every circle is an overtone"),
    (40.4, 42.9, "换一组圆，就换一种波形", "A different set of circles, a different wave"),
    (43.4, 47.6, "甚至，是一次心跳", "Even a heartbeat"),
    (48.6, 52.6, "让圆在平面上旋转，它们能画出一切", "Let circles spin in a plane, and they can draw anything"),
    (53.0, 58.8, "圆越多，画得越精确", "The more circles, the finer the drawing"),
    (60.6, 63.6, "现在，用 800 个圆……", "Now, with 800 circles..."),
    (75.6, 79.2, "……写下他的名字", "...write his name"),
]

CAPTIONS += [
    (80.6, 83.9, "声音，是空气的振动", "Sound is vibrating air"),
    (84.2, 87.8, "傅里叶变换，把混合的声音拆成一个个音符", "The Fourier transform unmixes it into notes"),
    (88.4, 91.8, "你的耳蜗里，就住着一台傅里叶分析仪", "Inside your ear lives a Fourier analyser"),
    (92.0, 95.7, "螺旋上的每一处，只对一个频率起舞", "Each point along the spiral dances to one frequency"),
    (96.6, 101.6, "每一首歌，都是一片频率的山脉", "Every song is a landscape of frequencies"),
    (104.4, 110.8, "这就是你此刻听到的音乐的频谱", "This is the live spectrum of the music you are hearing"),
]

CAPTIONS += [
    (112.6, 116.4, "图像，同样由波组成", "Images are made of waves too"),
    (116.8, 120.6, "按重要性，一个接一个地叠加二维波纹", "Add 2D ripples one by one, the most important first"),
    (121.0, 124.4, "只用 1% 的波，你已经能认出它", "With just 1% of the waves, you already know who it is"),
    (124.8, 127.8, "全部叠加，分毫不差", "Add them all, and it is perfect"),
    (128.4, 131.9, "去掉高频：只剩模糊", "Remove the high frequencies: only blur remains"),
    (132.3, 135.8, "去掉低频：只剩轮廓", "Remove the low frequencies: only edges remain"),
    (136.4, 139.9, "JPEG：手机里的每一张照片，都由这 64 种波纹拼成", "JPEG: every photo on your phone is built from these 64 patterns"),
    (140.2, 143.4, "丢掉人眼看不出的高频，文件就小了十倍", "Drop the frequencies your eye can't see, and the file shrinks tenfold"),
]

CAPTIONS += [
    (148.3, 153.8, "光谱中的暗线，揭示了太阳由什么构成", "Dark lines in sunlight reveal what the Sun is made of"),
    (154.3, 159.8, "1952 年，一张衍射照片让人类第一次看见双螺旋", "In 1952, a diffraction photo revealed the double helix"),
    (160.3, 165.8, "傅里叶逆变换，把频率还原成你的大脑", "An inverse Fourier transform turns frequencies into your brain"),
    (166.3, 171.8, "13 亿年前两个黑洞相撞，时空传来一声“啁啾”", "The chirp of two black holes that collided 1.3 billion years ago"),
    (172.3, 177.8, "成百上千个频率并肩传输，互不干扰", "Hundreds of frequencies carry data side by side, without interfering"),
    (178.3, 183.8, "位置越确定，动量越模糊——这正是傅里叶变换的性质", "The sharper the position, the blurrier the momentum"),
    (184.3, 189.8, "大模型用不同频率的正弦波，记住每个字的位置", "Language models mark each word's position with sine waves"),
    (190.3, 195.8, "从 N² 到 N log N：让这一切在你的手机里实时发生", "From N² to N log N: all of this, in real time, in your pocket"),
]

CAPTIONS += [
    (196.5, 199.8, "而这一切，始于 1807 年", "And it all began in 1807"),
    (200.1, 203.8, "傅里叶只是想弄明白：热，是如何传播的", "Fourier only wanted to understand how heat spreads"),
    (204.3, 207.8, "他没有想到，他给了人类一双新的眼睛", "He never knew he was giving us a new pair of eyes"),
    (208.1, 211.7, "一双，看见频率的眼睛", "Eyes that see frequencies"),
]

CHAPTERS = [
    (24.2, 47.8, "01", "圆 与 波", "CIRCLES & WAVES"),
    (48.2, 79.4, "02", "以 圆 作 画", "DRAWING WITH CIRCLES"),
    (80.2, 111.6, "03", "声 音", "SOUND"),
    (112.2, 143.6, "04", "图 像", "IMAGES"),
]


def overlays():
    return [ChapterTag(CHAPTERS), Captions(CAPTIONS)]
