#!/usr/bin/env python3
"""Draw the decode-speed bar chart of the README: one SVG for light pages, one for dark.

usage: make_decode_chart.py <out dir>

The numbers are the medians of 13 November 2026 reported in README.md of this directory (the grey bar: 12 November).
Edit ROWS, run, commit both files.
"""
import sys
from pathlib import Path

TITLE = 'DeepSeek V4.1 Flash, October checkpoint: steady decode in tokens per second'
NOTE = 'MacBook Pro M5 Max, 128 GB · 512-token prompt, 200 generated tokens · same logits in every row · × against the grey bar'
# label, tok/s, kind ("ref" = the engine without this fork's switches)
ROWS = [
    ('ds4 without the switches, internal SSD', 13.0, 'ref'),
    ('this fork, internal SSD', 18.38, 'fork'),
    ('+ one Thunderbolt 5 NVMe drive', 20.69, 'fork'),
    ('+ two Thunderbolt 5 NVMe drives', 22.47, 'fork'),
]
THEMES = {
    'light': {'text': '#0b0b0b', 'muted': '#52514e', 'bar': '#2a78d6', 'ref': '#85847e', 'axis': '#d9d8d2'},
    'dark': {'text': '#f0efec', 'muted': '#c3c2b7', 'bar': '#3987e5', 'ref': '#8f8e87', 'axis': '#3a3a37'},
}
W, X0, FULL, BAR, STEP, TOP = 760, 286, 340, 20, 38, 62
FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"


def bar(x, y, w, h, r):
    """Square at the baseline, rounded at the data end."""
    r = min(r, w)
    return ('M%g %g h%g a%g %g 0 0 1 %g %g v%g a%g %g 0 0 1 %g %g h%g z'
            % (x, y, w - r, r, r, r, r, h - 2 * r, r, r, -r, r, -(w - r)))


def svg(theme):
    c = THEMES[theme]
    top = max(v for _, v, _ in ROWS)
    base = next(v for _, v, k in ROWS if k == 'ref')
    height = TOP + STEP * len(ROWS) + 6
    desc = '; '.join('%s %.1f' % (label, v) for label, v, _ in ROWS)
    out = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
           'font-family="%s" aria-label="%s: %s.">' % (W, height, W, height, FONT, TITLE, desc),
           '<title>%s</title>' % TITLE,
           '<text x="0" y="18" font-size="15" font-weight="600" fill="%s">%s</text>' % (c['text'], TITLE),
           '<text x="0" y="38" font-size="12" fill="%s">%s</text>' % (c['muted'], NOTE),
           '<line x1="%g" y1="%d" x2="%g" y2="%d" stroke="%s" stroke-width="1"/>'
           % (X0 - 0.5, TOP - 6, X0 - 0.5, TOP + STEP * len(ROWS) - 12, c['axis'])]
    for i, (label, v, kind) in enumerate(ROWS):
        y = TOP + STEP * i
        w = FULL * v / top
        out.append('<text x="%d" y="%d" font-size="13" text-anchor="end" fill="%s">%s</text>'
                   % (X0 - 12, y + 15, c['muted'], label))
        out.append('<path d="%s" fill="%s"/>' % (bar(X0, y, w, BAR, 4), c['ref' if kind == 'ref' else 'bar']))
        out.append('<text x="%g" y="%d" font-size="14" font-weight="700" fill="%s">%.1f</text>'
                   % (X0 + w + 8, y + 15, c['text'], v))
        if kind != 'ref':
            out.append('<text x="%g" y="%d" font-size="12" fill="%s">%.2f×</text>'
                       % (X0 + w + 48, y + 15, c['muted'], v / base))
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for theme in THEMES:
        (out / ('decode-drives-%s.svg' % theme)).write_text(svg(theme), encoding='utf-8')
        print(out / ('decode-drives-%s.svg' % theme))


if __name__ == '__main__':
    main()
