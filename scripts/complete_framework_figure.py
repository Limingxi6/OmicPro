from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageDraw, ImageFont


ROOT = Path(r"E:\OmicMAP")
SOURCE = Path(r"C:\Users\Lenovo\Downloads\模型框架图3.pdf")
OUTPUT = ROOT / "模型框架图3_补充版.png"
SCALE = 1.5


def font(size, bold=False):
    name = "timesbd.ttf" if bold else "times.ttf"
    return ImageFont.truetype(rf"C:\Windows\Fonts\{name}", size)


def centered(draw, box, text, size, fill, bold=False):
    draw.text(
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        text,
        font=font(size, bold), fill=fill, anchor="mm",
    )


def arrow(draw, start, end, fill=(55, 55, 55), width=4):
    draw.line([start, end], fill=fill, width=width)
    x0, y0 = start
    x1, y1 = end
    if abs(x1 - x0) >= abs(y1 - y0):
        sign = 1 if x1 >= x0 else -1
        tri = [(x1, y1), (x1 - sign * 14, y1 - 7), (x1 - sign * 14, y1 + 7)]
    else:
        sign = 1 if y1 >= y0 else -1
        tri = [(x1, y1), (x1 - 7, y1 - sign * 14), (x1 + 7, y1 - sign * 14)]
    draw.polygon(tri, fill=fill)


def main():
    document = pdfium.PdfDocument(SOURCE)
    page = document[0]
    image = page.render(scale=SCALE).to_pil().convert("RGB")
    draw = ImageDraw.Draw(image)

    # This compact addition uses the paper's existing pale-orange loss nodes,
    # black outlines, dashed total-loss box, and thin orthogonal arrows.
    aux = (1450, 934, 1669, 1014)
    total = (1432, 1047, 1777, 1113)
    outline = (56, 56, 56)
    orange = (252, 227, 188)
    cream = (255, 253, 247)

    centered(draw, (1435, 900, 1700, 930), "Auxiliary Loss", 24, outline, True)
    draw.rounded_rectangle(aux, radius=20, fill=orange, outline=outline, width=3)
    centered(draw, aux, "Laux", 28, outline, True)

    # Dashed, compact total objective, matching the formula box already used in D.
    draw.rounded_rectangle(total, radius=12, fill=cream, outline=outline, width=3)
    for x in range(total[0] + 9, total[2] - 8, 13):
        draw.line((x, total[1], min(x + 7, total[2] - 8), total[1]), fill=outline, width=2)
        draw.line((x, total[3], min(x + 7, total[2] - 8), total[3]), fill=outline, width=2)
    for y in range(total[1] + 9, total[3] - 8, 13):
        draw.line((total[0], y, total[0], min(y + 7, total[3] - 8)), fill=outline, width=2)
        draw.line((total[2], y, total[2], min(y + 7, total[3] - 8)), fill=outline, width=2)
    centered(draw, total, "L = Lmain + λ Laux", 25, outline, True)

    # Fusion output branches to the reconstruction loss; prediction contributes Lmain.
    arrow(draw, (1401, 975), (1442, 975), width=3)
    arrow(draw, (1560, 902), (1560, 1038), width=3)
    arrow(draw, (1559, 1018), (1559, 1038), width=3)
    draw.line((1560, 1038, 1602, 1038), fill=outline, width=3)

    image.save(OUTPUT, quality=95)
    print(OUTPUT)


if __name__ == "__main__":
    main()
