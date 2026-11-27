# 用 LibreOffice（UNO）打开 .docx 副本，更新目录后导出 PDF；不回写 .docx
# 用法：<LibreOffice 自带 python> export_pdf.py <soffice 路径> <输入 .docx> <输出 .pdf>
# 需预先设置 FONTCONFIG_FILE 指向字体映射配置
import sys, os, time, shutil, subprocess, tempfile
import uno
from com.sun.star.beans import PropertyValue
from com.sun.star.connection import NoConnectException


def prop(name, value):
    p = PropertyValue()
    p.Name, p.Value = name, value
    return p


SOFFICE, SRC, OUT = sys.argv[1:4]
work = tempfile.mkdtemp(prefix="kaiti_pdf_")
src_copy = os.path.join(work, "input.docx")
shutil.copyfile(SRC, src_copy)
pipe = f"kaiti_{os.getpid()}"
proc = subprocess.Popen([SOFFICE, "--headless", "--invisible", "--nologo", "--norestore", "--nodefault",
                         f"-env:UserInstallation=file://{work}/profile", f"--accept=pipe,name={pipe};urp;"])
try:
    local = uno.getComponentContext()
    resolver = local.ServiceManager.createInstanceWithContext("com.sun.star.bridge.UnoUrlResolver", local)
    for _ in range(120):
        try:
            ctx = resolver.resolve(f"uno:pipe,name={pipe};urp;StarOffice.ComponentContext")
            break
        except NoConnectException:
            time.sleep(0.5)
    else:
        raise RuntimeError("LibreOffice did not start")
    desktop = ctx.ServiceManager.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
    doc = desktop.loadComponentFromURL(uno.systemPathToFileUrl(src_copy), "_blank", 0, (prop("Hidden", True),))
    indexes = doc.getDocumentIndexes()
    print("indexes:", indexes.getCount())
    # 更新两次：第一次更新可能改变目录长度，从而影响后续页码
    for _ in range(2):
        for i in range(indexes.getCount()):
            indexes.getByIndex(i).update()
        doc.getTextFields().refresh()
        doc.refresh()
    doc.storeToURL(uno.systemPathToFileUrl(os.path.abspath(OUT)),
                   (prop("FilterName", "writer_pdf_Export"),
                    prop("FilterData", uno.Any("[]com.sun.star.beans.PropertyValue", tuple([
                        prop("ExportBookmarks", True),        # PDF 书签（目录导航）
                        prop("UseTaggedPDF", True),
                    ])))))
    doc.close(True)
    print("wrote", OUT)
finally:
    try:
        desktop.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
    shutil.rmtree(work, ignore_errors=True)
