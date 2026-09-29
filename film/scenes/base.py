"""Scene base class and overlay layers (captions, chapter tags)."""
from ..core import (GOLD, H, W, WHITE, clamp, draw_text, draw_text_chars, ease_out, set_rgba, smooth)


class Scene:
    start = 0.0
    end = 0.0
    fade_in = 0.0   # crossfade length before `start`
    fade_out = 0.0  # crossfade length after `end`
    bloom = 0.9

    def prepare(self):
        pass

    def weight(self, t):
        if t < self.start:
            return smooth((t - (self.start - self.fade_in)) / self.fade_in) if self.fade_in > 0 else 0.0
        if t >= self.end:
            return 1 - smooth((t - self.end) / self.fade_out) if self.fade_out > 0 else 0.0
        return 1.0

    def draw(self, cv, t, lt):
        raise NotImplementedError

    def effects(self, t, lt):
        return {}


class Captions:
    """Bottom-centred bilingual captions with per-character reveal."""

    def __init__(self, items):
        # items: (t0, t1, zh, en)
        self.items = items

    def draw(self, cv, t):
        ctx = cv.ctx
        for t0, t1, zh, en in self.items:
            if t < t0 or t > t1 + 0.6:
                continue
            lt = t - t0
            a_out = 1 - smooth((t - t1) / 0.45)
            y = 948
            # soft dark band for legibility
            band = min(1.0, lt / 0.4) * a_out
            if band > 0:
                import cairo
                g = cairo.LinearGradient(0, y - 90, 0, H)
                g.add_color_stop_rgba(0, 0, 0, 0, 0)
                g.add_color_stop_rgba(0.5, 0, 0, 0, 0.35 * band)
                g.add_color_stop_rgba(1, 0, 0, 0, 0.45 * band)
                ctx.set_source(g)
                ctx.rectangle(0, y - 90, W, H - y + 90)
                ctx.fill()
            draw_text_chars(ctx, zh, W / 2, y, lt, size=42, font="sans", weight=500, color=WHITE, alpha=0.96 * a_out,
                            tracking=0.08, stagger=min(0.045, 0.9 / max(1, len(zh))), dur=0.45, rise=14)
            if en:
                a = smooth((lt - 0.35) / 0.6) * a_out * 0.62
                draw_text(ctx, en.upper(), W / 2, y + 50, size=17, font="latin", weight=400, color=(0.8, 0.9, 1.0),
                          alpha=a, tracking=0.22)


class ChapterTag:
    """Top-left chapter label: big thin number + zh title + en subtitle."""

    def __init__(self, items):
        # (t0, t1, number, zh, en)
        self.items = items

    def draw(self, cv, t):
        ctx = cv.ctx
        for t0, t1, num, zh, en in self.items:
            if t < t0 or t > t1:
                continue
            lt = t - t0
            a = smooth(lt / 0.8) * (1 - smooth((t - (t1 - 0.6)) / 0.6))
            # intro emphasis for the first 3.5 s, then settle to a quieter state
            emph = 1 - smooth((lt - 3.0) / 1.0)
            x, y = 96, 92
            draw_text(ctx, num, x, y, size=54, font="latin", weight=200, color=GOLD, alpha=a * (0.55 + 0.4 * emph),
                      anchor="left", tracking=0.05)
            set_rgba(ctx, GOLD, a * 0.6)
            ctx.set_line_width(1.2)
            L = 40 + 60 * ease_out(lt / 1.2)
            ctx.move_to(x + 82, y)
            ctx.line_to(x + 82 + L, y)
            ctx.stroke()
            draw_text(ctx, zh, x + 82 + L + 18, y - 11, size=26, font="serif", weight=600, color=WHITE,
                      alpha=a * (0.55 + 0.4 * emph), anchor="left", tracking=0.18)
            draw_text(ctx, en, x + 82 + L + 20, y + 20, size=13, font="latin", weight=500,
                      color=(0.75, 0.85, 1.0), alpha=a * (0.35 + 0.35 * emph), anchor="left", tracking=0.35)
