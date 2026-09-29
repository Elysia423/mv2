# 万物皆波 · 傅里叶变换宣传片

一部约 3 分 52 秒的 1080p 宣传片：从一个旋转的圆出发，讲傅里叶变换怎样把声音、图像、光、DNA、大脑、黑洞、Wi-Fi、量子和 AI 串在一起。

**每一帧画面都由代码实时计算生成，每一个声音都由正弦波叠加合成**：配乐、鼓点和噪声全是正弦波（噪声用随机相位的逆 FFT 生成）。所以片中的频谱可视化，显示的就是你正在听的那段配乐的真实频谱。

## 分镜

| 时间 | 段落 | 内容 |
| --- | --- | --- |
| 0:00 | 冷开场 | 1807 年，一个圆投影出正弦波；“拉格朗日：不可能。” |
| 0:16 | 片名 | 傅里叶 · 万物，皆是波的叠加 |
| 0:24 | 01 圆与波 | 1→60 个圆逼近方波（**配乐中的音色跟着圆的数量一起变化**），接着是锯齿波，再到心跳（心电图 + 心跳声） |
| 0:48 | 02 以圆作画 | 8 / 24 / 80 / 400 个圆画高音谱号；800 个圆用毛笔字写出“傅里叶”，镜头跟着笔尖推进 |
| 1:20 | 03 声音 | 三个纯音混成一个和弦，再拆开落到琴键上；耳蜗频率螺旋；配乐的实时频谱山脉和环形频谱 |
| 1:52 | 04 图像 | 按重要性逐个叠加二维波纹重建一张猫的照片（1% 已可辨认）；低通 / 高通滤波；JPEG 的 64 种 DCT 波纹 |
| 2:24 | 不止于此 | 棱镜与太阳光谱暗线 · DNA 与 Photo 51 · 核磁共振 k 空间 · 引力波“啁啾” · Wi-Fi/5G 的 OFDM · 量子不确定性 · Transformer 位置编码 · FFT 蝶形网络 |
| 3:16 | 起源 | 1807 年的热传导问题：高频先衰减，温度越来越平滑 |
| 3:24 | 蒙太奇 | 随节拍快切全片镜头 |
| 3:32 | 终章 | 傅里叶变换 · 换一个角度，看见万物的频率 · 一切，从一个圆开始 |

## 构建

依赖：Python 3.11+，以及 `numpy scipy pycairo opencv-python-headless pillow matplotlib fonttools scikit-image pooch imageio-ffmpeg`。

```bash
pip install numpy scipy pycairo opencv-python-headless pillow matplotlib fonttools scikit-image pooch imageio-ffmpeg
./fetch_assets.sh                      # 字体（Google Fonts）+ 脑部 MRI 样例数据
python -m film.render audio            # 合成配乐 build/soundtrack.wav 和逐帧频谱
python -m film.render still 70 157     # 预览指定时刻的静帧 -> build/still_*.png
python -m film.render video            # 渲染成片 -> build/fourier.mp4（4 核约 20 分钟）
python -m film.render video --scale 0.5 --from 48 --to 80   # 低分辨率预览某一段
```

## 代码结构

- `film/story.py`：全片时间轴。节拍、和弦、旋律、第一章“圆的数量”调度，**画面和声音共用这一份**。
- `film/audio.py`：加法合成器与编曲（pad、拨弦、贝斯、鼓、引力波啁啾、心跳），外加卷积混响和限幅器。
- `film/core.py`：cairo 画布、文字排版、缓动、多尺度泛光（bloom）、暗角、颗粒等后期。
- `film/epicycles.py`：把字形轮廓变成闭合路径，再用 FFT 求出本轮（epicycle）系数。
- `film/scenes/`：各个段落，字幕在 `scenes/__init__.py`。
- `film/render.py`：多进程逐帧渲染，结果通过管道交给 ffmpeg 编码。
