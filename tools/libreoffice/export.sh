#!/bin/bash
# 用本目录中的 LibreOffice 26.2.6.3（TDF 官方 deb 解包，未安装到系统）把 .docx 导出为 PDF，导出前更新目录
# 用法：tools/libreoffice/export.sh <输入.docx> <输出.pdf>
# 字体：宋体→FandolSong、黑体→FandolHei、华文新魏→用户提供的 STXINWEI、Times New Roman（fonts/fonts.conf）
D="$(cd "$(dirname "$0")" && pwd)"
export FONTCONFIG_FILE="$D/fonts/fonts.conf"
exec "$D/root/opt/libreoffice26.2/program/python" "$D/export_pdf.py" "$D/root/opt/libreoffice26.2/program/soffice" "$1" "$2"
