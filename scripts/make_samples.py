"""造一套多格式测试数据：一个虚构项目「星河储能」的资料包（docx / pptx / xlsx / 各种图片 / PDF），
带目录层级，打成 samples/xinghe.zip；同时把单个文件放在 samples/formats/ 下方便逐个测。
每份文件里埋了可核对的事实，问题和标准答案在 scripts/multiformat_cases.json。

  .venv/bin/python scripts/make_samples.py
需要 python-docx、python-pptx、openpyxl（requirements-dev.txt）。
"""

import io
import shutil
import zipfile
from pathlib import Path

import pymupdf
from docx import Document

from openpyxl import Workbook
from openpyxl.styles import Font
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt as PPt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "xinghe"          # folder tree that becomes the zip
FLAT = ROOT / "samples" / "formats"         # one file per format


def filler(topic: str, n: int = 3) -> list[str]:
    """Plausible but fact-free paragraphs so documents have some bulk."""
    base = [
        f"围绕{topic}，项目组在前期调研中梳理了现有方案的不足，并结合星河储能的产线条件给出了改进方向。",
        f"{topic}的实施需要与运维、采购、安全等部门协同推进，相关接口人和职责在附件中列明。",
        f"在{topic}方面，我们参考了行业通行做法和国家标准，并保留了后续扩展的余地。",
        f"针对{topic}可能出现的风险，项目组制定了分级响应预案，并安排了季度复盘。",
    ]
    return [base[i % len(base)] for i in range(n)]


# --------------------------------------------------------------------------- docx

def proposal_docx(path):
    d = Document()
    d.add_heading("星河储能 2025 年度工商业储能项目建议书", 0)
    d.add_heading("第一章 项目背景", 1)
    for p in filler("工商业储能市场"):
        d.add_paragraph(p)
    d.add_paragraph("长三角地区两部制电价下，峰谷价差已达到 0.82 元/kWh，为用户侧储能提供了套利空间。")
    d.add_heading("第二章 建设内容", 1)
    d.add_heading("2.1 建设规模", 2)
    d.add_paragraph("项目计划在苏州、无锡、常州三地建设 12 座用户侧储能电站，总装机 48 MW / 96 MWh。")
    for p in filler("建设规模"):
        d.add_paragraph(p)
    d.add_heading("2.2 选址原则", 2)
    for p in filler("选址"):
        d.add_paragraph(p)
    d.add_paragraph("单站接入容量不低于 2 MW，且用户年用电量需超过 800 万 kWh。")
    d.add_heading("第三章 投资估算", 1)
    d.add_paragraph("项目总投资 1.38 亿元，其中设备购置费 1.02 亿元，建安工程费 0.21 亿元，其他费用 0.15 亿元。")
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, ["费用项", "金额（万元）", "占比"]):
        c.text = h
    for row in [("设备购置费", "10200", "73.9%"), ("建安工程费", "2100", "15.2%"), ("其他费用", "1500", "10.9%")]:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v
    d.add_heading("第四章 经济效益", 1)
    d.add_paragraph("测算全投资内部收益率（IRR）为 9.6%，静态投资回收期 6.8 年。")
    for p in filler("经济效益"):
        d.add_paragraph(p)
    d.add_heading("第五章 风险与对策", 1)
    for p in filler("政策与电价风险", 4):
        d.add_paragraph(p)
    d.save(path)


def tech_docx(path):
    d = Document()
    d.add_heading("储能电站技术方案", 0)
    sections = [
        ("1 系统架构", [("1.1 总体架构", "系统采用「电池舱 + PCS 舱 + EMS」三层架构，单舱容量 3.72 MWh。"),
                       ("1.2 通信方案", "站内通信采用 IEC 61850，站控层与云平台之间使用 MQTT over TLS。")]),
        ("2 电池系统", [("2.1 电芯选型", "选用 314Ah 磷酸铁锂电芯，循环寿命不低于 8000 次（70% SOH）。"),
                       ("2.2 电池管理系统", "BMS 采用三级架构，单体电压采样精度 ±1 mV。"),
                       ("2.3 热管理", "采用液冷方案，舱内电芯最大温差控制在 3 ℃ 以内。")]),
        ("3 变流与并网", [("3.1 PCS 配置", "每座电站配置 2 台 1250 kW 储能变流器，效率不低于 98.5%。"),
                        ("3.2 并网要求", "满足 GB/T 36547-2018，具备低电压穿越能力。")]),
        ("4 消防与安全", [("4.1 消防系统", "每个电池舱配置全氟己酮气体灭火与舱级水消防两级防护。"),
                        ("4.2 安全监测", "部署可燃气体（H2、CO）探测器，报警阈值为 50 ppm。")]),
        ("5 运维方案", [("5.1 运维组织", "采用「区域中心 + 无人值守」模式，苏州设区域运维中心。"),
                       ("5.2 巡检周期", "日常远程巡检每天 2 次，现场巡检每月 1 次。")]),
    ]
    for h1, subs in sections:
        d.add_heading(h1, 1)
        d.add_paragraph(filler(h1[2:])[0])
        for h2, fact in subs:
            d.add_heading(h2, 2)
            d.add_paragraph(fact)
            for p in filler(h2.split(" ", 1)[1], 4):
                d.add_paragraph(p)
    d.save(path)


def english_docx(path):
    d = Document()
    d.add_heading("Supplier Evaluation Memo — Battery Cells", 0)
    d.add_heading("Summary", 1)
    d.add_paragraph("Three cell suppliers were evaluated in March 2025. Supplier B (Hengtai Energy) ranked first "
                    "with a weighted score of 87.5 out of 100.")
    d.add_heading("Criteria", 1)
    d.add_paragraph("Weights: price 40%, cycle life 30%, delivery 20%, after-sales service 10%.")
    d.add_heading("Results", 1)
    d.add_paragraph("Supplier A scored 81.0, Supplier C scored 76.5. The quoted price of Supplier B was "
                    "0.38 CNY per Wh, valid until 30 June 2025.")
    d.save(path)


# --------------------------------------------------------------------------- pptx

def review_pptx(path):
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)

    def slide(title, bullets):
        s = prs.slides.add_slide(prs.slide_layouts[1])
        s.shapes.title.text = title
        tf = s.placeholders[1].text_frame
        tf.text = bullets[0]
        for b in bullets[1:]:
            tf.add_paragraph().text = b
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.size = PPt(20)

    s = prs.slides.add_slide(prs.slide_layouts[0])
    s.shapes.title.text = "星河储能项目立项评审会"
    s.placeholders[1].text = "2025 年 2 月 18 日 · 投资决策委员会"
    slide("评审议程", ["项目概况", "技术路线", "投资与收益", "评审结论"])
    slide("项目概况", ["12 座用户侧储能电站，总装机 48 MW / 96 MWh", "首批 4 座电站 2025 年 6 月并网"])
    slide("技术路线", ["314Ah 磷酸铁锂电芯 + 液冷", "PCS 效率 ≥ 98.5%", "EMS 自研，支持需量管理"])
    slide("投资与收益", ["总投资 1.38 亿元", "IRR 9.6%，回收期 6.8 年"])
    slide("评审结论", ["评审委员 7 人，6 票同意、1 票弃权，项目通过立项",
                       "要求：首批电站投运后 3 个月内提交运行评估报告"])
    prs.save(path)


# --------------------------------------------------------------------------- xlsx

def budget_xlsx(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "年度预算汇总"
    ws.append(["预算科目", "Q1", "Q2", "Q3", "Q4", "全年合计（万元）"])
    rows = [("设备采购", 1800, 3600, 2900, 1900), ("工程建设", 300, 700, 650, 450),
            ("人力成本", 210, 220, 230, 240), ("运维费用", 40, 80, 120, 160), ("市场推广", 60, 90, 70, 80)]
    for name, *q in rows:
        ws.append([name, *q, sum(q)])
    ws.append(["合计", *[sum(r[i + 1] for r in rows) for i in range(4)], sum(sum(r[1:]) for r in rows)])
    for c in ws[1]:
        c.font = Font(bold=True)
    ws2 = wb.create_sheet("电站清单")
    ws2.append(["编号", "城市", "用户", "装机（MW）", "容量（MWh）", "计划并网"])
    sites = [("XH-01", "苏州", "苏州精密机械有限公司", 6, 12, "2025-06"),
             ("XH-02", "苏州", "吴江纺织产业园", 4, 8, "2025-06"),
             ("XH-03", "无锡", "无锡半导体封测厂", 5, 10, "2025-06"),
             ("XH-04", "常州", "常州新能源汽车零部件厂", 3, 6, "2025-06"),
             ("XH-05", "无锡", "江阴港口物流中心", 4, 8, "2025-09"),
             ("XH-06", "苏州", "昆山电子信息产业园", 5, 10, "2025-09"),
             ("XH-07", "常州", "常州钢构制造基地", 3, 6, "2025-09"),
             ("XH-08", "无锡", "宜兴陶瓷产业园", 4, 8, "2025-12"),
             ("XH-09", "苏州", "太仓化工新材料园", 4, 8, "2025-12"),
             ("XH-10", "常州", "溧阳动力电池基地", 4, 8, "2025-12"),
             ("XH-11", "无锡", "无锡数据中心", 3, 6, "2026-03"),
             ("XH-12", "苏州", "常熟服装城", 3, 6, "2026-03")]
    for s in sites:
        ws2.append(list(s))
    ws3 = wb.create_sheet("说明")
    ws3.append(["本预算由财务部于 2025 年 1 月编制，经总经理办公会审议通过。预备费按设备采购的 3% 计提，未计入上表。"])
    wb.save(path)


def ops_xlsx(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "2025Q3 运行数据"
    ws.append(["电站", "充电量（MWh）", "放电量（MWh）", "综合效率", "可用率", "收益（万元）"])
    data = [("XH-01", 1040, 905, "87.0%", "99.2%", 74.2), ("XH-02", 690, 597, "86.5%", "98.7%", 48.9),
            ("XH-03", 870, 761, "87.5%", "99.5%", 62.4), ("XH-04", 515, 443, "86.0%", "97.9%", 36.3)]
    for r in data:
        ws.append(list(r))
    ws.append(["合计", 3115, 2706, "86.9%", "", 221.8])
    wb.save(path)


# --------------------------------------------------------------------------- images

def text_image(lines: list[str], w=1000, h=640, title_size=34, size=24) -> Image.Image:
    """Render text to an image (CJK via PyMuPDF's built-in font) so OCR has something to read."""
    doc = pymupdf.open()
    page = doc.new_page(width=w, height=h)
    page.draw_rect(page.rect, color=None, fill=(0.98, 0.98, 0.96))
    y = 70
    for i, line in enumerate(lines):
        fs = title_size if i == 0 else size
        page.insert_text((60, y), line, fontname="china-s", fontsize=fs, color=(0.1, 0.1, 0.15))
        y += fs * 1.9
    pix = page.get_pixmap(dpi=110)
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


def images(folder: Path):
    arch = text_image(["储能电站系统架构图（说明）", "云平台  ←MQTT over TLS→  站控层 EMS",
                       "站控层 EMS  ←IEC 61850→  PCS 舱（2 × 1250 kW）", "PCS 舱  ←→  电池舱（单舱 3.72 MWh，液冷）",
                       "图注：本图为 XH-01 苏州精密机械站，2025 年 6 月版"])
    arch.save(folder / "系统架构图.png")
    plate = text_image(["设备铭牌", "产品型号：XH-ESS-3720L", "额定容量：3720 kWh", "额定电压：1331.2 V",
                        "防护等级：IP55", "制造日期：2025-04-12", "出厂编号：XH25040086"], w=800, h=560)
    plate.save(folder / "电池舱铭牌.jpg", quality=88)
    notice = text_image(["现场安全告示", "1. 进入电池舱前须确认 H2/CO 浓度低于 50 ppm", "2. 动火作业须办理一级动火证",
                         "3. 紧急联系人：运维值班 0512-6688 0000"], w=900, h=480)
    notice.save(folder / "安全告示.webp", quality=90)
    return {"png": folder / "系统架构图.png", "jpg": folder / "电池舱铭牌.jpg", "webp": folder / "安全告示.webp",
            "_img": plate}


# --------------------------------------------------------------------------- main

def main():
    shutil.rmtree(OUT, ignore_errors=True)
    shutil.rmtree(FLAT, ignore_errors=True)
    dirs = {k: OUT / k for k in ["01_立项", "02_财务", "03_技术/图纸", "03_技术/现场照片", "04_采购", "05_参考资料"]}
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    FLAT.mkdir(parents=True)

    proposal_docx(dirs["01_立项"] / "项目建议书.docx")
    review_pptx(dirs["01_立项"] / "立项评审会.pptx")
    budget_xlsx(dirs["02_财务"] / "2025年度预算.xlsx")
    ops_xlsx(dirs["02_财务"] / "2025Q3运行数据.xlsx")
    tech_docx(OUT / "03_技术" / "技术方案.docx")
    imgs = images(dirs["03_技术/现场照片"])
    shutil.move(imgs["png"], dirs["03_技术/图纸"] / "系统架构图.png")
    english_docx(dirs["04_采购"] / "supplier-evaluation.docx")
    src = ROOT / "samples" / "earthmover.pdf"
    if src.exists():
        shutil.copy(src, dirs["05_参考资料"] / "earthmover.pdf")
    (OUT / "README.txt").write_text("资料包说明：不支持的格式应被跳过。\n", encoding="utf-8")

    # one of each format for single-file tests (gif / bmp / jp2 from the nameplate image)
    for f in OUT.rglob("*"):
        if f.suffix.lower() in (".docx", ".pptx", ".xlsx", ".png", ".jpg", ".webp"):
            shutil.copy(f, FLAT / f.name)
    plate = imgs["_img"]
    plate.save(FLAT / "铭牌.gif")
    plate.save(FLAT / "铭牌.bmp")
    try:
        plate.save(FLAT / "铭牌.jp2")
    except (OSError, KeyError) as e:
        print("jp2 skipped:", e)

    # the zip, with junk a Mac would add, to check filtering
    zpath = ROOT / "samples" / "xinghe.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(OUT.rglob("*")):
            if f.is_file():
                z.write(f, Path("星河储能项目资料") / f.relative_to(OUT))
        z.writestr("__MACOSX/星河储能项目资料/._README.txt", b"\0")
        z.writestr("星河储能项目资料/.DS_Store", b"\0")
    print("生成", zpath, "和", FLAT)
    for f in sorted(FLAT.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size:>8d}")


if __name__ == "__main__":
    main()
