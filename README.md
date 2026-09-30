# 万物皆波 · 傅里叶变换宣传片

一部 3 分 46 秒的 1080p 宣传片。从一个旋转的圆讲起，一路讲到声音、图像、光谱、DNA、大脑、黑洞、潮汐、Wi-Fi、量子、AI，以及拉普拉斯变换、小波等更大的“变换家族”。镜头 4–6 秒一切；数学概念尽量做成三维画面：欧拉螺旋、时域与频域之间的 90° 旋转、沿时间轴展开的本轮、三维频谱地形、球谐函数、s 平面、热方程曲面。

![stills](docs/stills.jpg)

成片：[`output/fourier.mp4`](output/fourier.mp4)（1920×1080 · 30 fps · 3:46）

**每一帧画面都由代码实时计算生成，每一个声音都由正弦波叠加合成**：配乐、鼓点和噪声全是正弦波（噪声用随机相位的逆 FFT 生成）。所以片中的频谱画面（耳蜗、听歌识曲、三维频谱、MP3 掩蔽），显示的就是你正在听的那段配乐的真实频谱。

## 分镜

| 时间 | 段落 | 内容 |
| --- | --- | --- |
| 0:00 | 冷开场 | 1807 年《论热的传播》；拉格朗日：“不可能。”15 年后才得以出版 |
| 0:12 | 片名 | 傅里叶变换 · 万物，皆是波的叠加 |
| 0:18 | 01 圆与波 | 欧拉公式的三维螺旋 → 方波（圆的个数就是你听到的泛音个数）→ 吉布斯现象的 9% 过冲 → 转过 90°，时域变成频域 → 锯齿波、心电图与 50 Hz 工频滤波 |
| 0:42 | 02 以圆作画 | 8/24/80/400 个圆逼近高音谱号；600 个圆画蝴蝶，轨迹沿时间轴展开成三维雕塑；DFT 公式与小提琴、DNA、大脑、地球；托勒密用本轮解释火星逆行 |
| 1:06 | 03 声音 | 和弦拆成三个音；“缠绕机”揭示傅里叶积分；耳蜗；听歌识曲的星图指纹；配乐的三维频谱地形；降噪耳机与 MP3 心理声学掩蔽 |
| 1:40 | 04 图像 | 二维波纹做成三维曲面；1% 的波纹就能认出猫；照片频谱的三维地形；低通与高通；JPEG 的 DCT |
| 2:10 | 不止于此 | 每 4 秒一个领域：氦的发现、韦布望远镜的衍射星芒、DNA Photo 51、MRI k 空间、引力波啁啾、球谐函数、开尔文潮汐预测机、OFDM、不确定性原理与 Shor 算法、Transformer 位置编码、FFT（以及高斯 1805 年的手稿） |
| 2:58 | 05 变换家族 | 拉普拉斯变换的三维 s 平面（虚轴切片就是傅里叶变换）；短时傅里叶与小波的时频分辨率对比 |
| 3:10 | 起源 | 热方程的三维曲面：高频最先衰减 |
| 3:18 | 蒙太奇 | 随节拍快切全片镜头 |
| 3:26 | 终章 | 傅里叶变换 · 换一个角度，看见万物的频率 · 一切，从一个圆开始 |

## 构建

依赖：Python 3.11+，以及 `numpy scipy pycairo opencv-python-headless pillow matplotlib fonttools scikit-image pooch imageio-ffmpeg`。

```bash
pip install numpy scipy pycairo opencv-python-headless pillow matplotlib fonttools scikit-image pooch imageio-ffmpeg
./fetch_assets.sh                      # 字体（Google Fonts）+ 脑部 MRI 样例数据
python -m film.render audio            # 合成配乐 build/soundtrack.wav 和逐帧频谱
python -m film.render still 70 157     # 预览指定时刻的静帧 -> build/still_*.png
python -m film.render video            # 渲染成片 -> build/fourier.mp4（4 核约 25 分钟）
python -m film.render video --scale 0.5 --from 42 --to 66   # 低分辨率预览某一段
```

## 代码结构

- `film/story.py`：全片时间轴。节拍、和弦、旋律、第一章“圆的数量”调度，**画面和声音共用这一份**。
- `film/three.py`：一个小型透视相机，外加三维折线、线框曲面和实体曲面（画家算法）的绘制工具。
- `film/audio.py`：加法合成器与编曲（pad、拨弦、贝斯、鼓、引力波啁啾、心跳），外加卷积混响和限幅器。
- `film/core.py`：cairo 画布、文字排版、缓动、多尺度泛光（bloom）、暗角、颗粒等后期。
- `film/epicycles.py`：把字形轮廓（高音谱号、Noto Emoji 的蝴蝶等）变成闭合路径，再用 FFT 求出本轮（epicycle）系数。
- `film/scenes/`：各个段落，字幕在 `scenes/__init__.py`。
- `film/render.py`：多进程逐帧渲染，结果通过管道交给 ffmpeg 编码。
