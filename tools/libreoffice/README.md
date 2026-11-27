# LibreOffice（仅用于导出预览 PDF）

- 版本 26.2.6.3，来源 https://downloadarchive.documentfoundation.org/libreoffice/old/26.2.6.3/deb/x86_64/
- 2026-11-26 下载：SHA-256 fd0e8f8f2408dd2e5b90286e60f3f97cf566ba441cd48cfc5bcc68067303e0bc 与官方镜像列表一致；
  GPG 签名有效（LibreOffice Build Team 代码签名密钥 C283 9ECA D940 8FBE 9531 C3E9 F434 A1EF AFEE AEA3）。
- 各 .deb 用 dpkg-deb -x 解包到 root/，没有安装到系统。原来放在 /tmp/lo，被系统的 /tmp 自动清理删去了部分文件，因此移到这里。
- 字体在 fonts/：Fandol（宋体、黑体的开源替代）、Times New Roman、用户提供的华文新魏；fonts.conf 只对导出生效。
- 用法：tools/libreoffice/export.sh <输入.docx> <输出.pdf>
