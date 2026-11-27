# 按 PDF 中图片的显示区域以 300 dpi 渲染并裁去白边，输出 figures/*.jpg 及显示尺寸
import sys, json, io
import pymupdf
from PIL import Image, ImageChops

PDF = sys.argv[1]
OUT = sys.argv[2]
DPI = 300
# (文件名, 页码(1 起), 图片 xref)
FIGS = [
    ("fig1", 3, 175),
    ("fig2a", 3, 184),
    ("fig2b", 3, 188),
    ("fig3", 10, 480),
    ("fig4", 11, 504),
    ("fig5", 12, 545),
    ("fig6", 12, 551),
]

doc = pymupdf.open(PDF)
sizes = {}
for name, pno, xref in FIGS:
    page = doc[pno - 1]
    info = next(i for i in page.get_image_info(xrefs=True) if i["xref"] == xref)
    rect = pymupdf.Rect(info["bbox"]) + (-1, -1, 1, 1)
    rect &= page.rect
    pix = page.get_pixmap(clip=rect, dpi=DPI, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    # 裁白边（阈值 245，避免 JPEG 噪点）
    diff = ImageChops.difference(img, Image.new("RGB", img.size, (255, 255, 255))).convert("L")
    bbox = diff.point(lambda v: 255 if v > 10 else 0).getbbox()
    img = img.crop(bbox)
    if name == "fig1":
        # 图1-1 底部嵌有乱码标签“(a) PC□□□”：截到最后一行绿色电路板像素为止
        def green_ratio(y):
            row = [img.getpixel((x, y)) for x in range(0, img.width, 3)]
            return sum(1 for r, g, b in row if g > 80 and g > r + 30) / len(row)
        last = max(y for y in range(img.height) if green_ratio(y) > 0.3)
        img = img.crop((0, 0, img.width, last + 1))
    img.save(f"{OUT}/{name}.jpg", quality=95, dpi=(DPI, DPI))
    w_pt, h_pt = img.width * 72 / DPI, img.height * 72 / DPI
    sizes[name] = {"px": [img.width, img.height], "pt": [round(w_pt, 1), round(h_pt, 1)]}
    print(name, sizes[name])
json.dump(sizes, open(f"{OUT}/sizes.json", "w"), indent=1)
