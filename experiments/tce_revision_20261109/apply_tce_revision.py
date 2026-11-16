# -*- coding: utf-8 -*-
"""
对《从跟踪到检测：CST Anti-UAV检测评测协议与时序输入.docx》执行 TCE 返修编辑（E1–E7）。
段落定位一律使用 strip 后的文本精确匹配；新标题/正文/表题均深拷贝原文档对应元素再改文字，
保证字体、缩进、加粗、边框与全文一致。运行一次即完成全部编辑并覆盖保存原文件。
"""
import copy
import docx
from lxml import etree

PATH = '/root/autodl-tmp/Ryan/cst_antiuav/从跟踪到检测：CST Anti-UAV检测评测协议与时序输入.docx'
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
W14 = 'http://schemas.microsoft.com/office/word/2010/wordml'
def qn(t): return '{%s}%s' % (W, t)
PARAID = '{%s}paraId' % W14
TEXTID = '{%s}textId' % W14

d = docx.Document(PATH)
body = d.element.body

# ---------- 通用工具 ----------
def para_text(p):
    return ''.join(t.text or '' for t in p.iter(qn('t')))

def strip_ids(el):
    for e in el.iter():
        for a in (PARAID, TEXTID):
            if a in e.attrib:
                del e.attrib[a]

def find_paras():
    return [ch for ch in body if ch.tag == qn('p')]

def get_anchor(desc, pred):
    hits = [p for p in find_paras() if pred(para_text(p).strip())]
    assert len(hits) == 1, 'anchor %s matched %d paragraphs' % (desc, len(hits))
    return hits[0]

def set_text(p, text):
    """重写段落文字：保留 pPr 与首 run 的 rPr，其余内容移除。用于标题改名与摘要重写。"""
    runs = p.findall(qn('r'))
    assert runs, 'paragraph has no run to rewrite'
    first = runs[0]
    for ch in list(p):
        if ch.tag == qn('pPr') or ch is first:
            continue
        p.remove(ch)
    for ch in list(first):
        if ch.tag != qn('rPr'):
            first.remove(ch)
    t = etree.SubElement(first, qn('t'))
    t.text = text

def clone_para(src_el, text):
    """深拷贝源段落（样式/缩进/字体完全一致），替换为单一 run 的给定文本。"""
    new = copy.deepcopy(src_el)
    strip_ids(new)
    for tag in ('bookmarkStart', 'bookmarkEnd'):
        for b in new.findall(qn(tag)):
            new.remove(b)
    for ch in list(new):
        if ch.tag != qn('pPr'):
            new.remove(ch)
    src_run = src_el.find(qn('r'))
    src_rpr = src_run.find(qn('rPr')) if src_run is not None else None
    r = etree.SubElement(new, qn('r'))
    if src_rpr is not None:
        r.append(copy.deepcopy(src_rpr))
    t = etree.SubElement(r, qn('t'))
    t.text = text
    return new

def make_table(template_tbl, rows):
    """深拷贝模板表格，保留表头行 + len(rows)-1 个数据行（删除多余行），并改写单元格文字。
    表级三线边框与表头单元格底边框随 deepcopy 保留。"""
    new = copy.deepcopy(template_tbl)
    strip_ids(new)
    trs = new.findall(qn('tr'))
    need = len(rows)
    for tr in trs[need:]:
        new.remove(tr)
    trs = new.findall(qn('tr'))
    assert len(trs) == need
    for ri, rowdata in enumerate(rows):
        tcs = trs[ri].findall(qn('tc'))
        assert len(tcs) == len(rowdata), 'row %d column mismatch' % ri
        for ci, txt in enumerate(rowdata):
            tc = tcs[ci]
            ps = tc.findall(qn('p'))
            for p in ps[1:]:
                tc.remove(p)
            p = ps[0]
            runs = p.findall(qn('r'))
            rpr = runs[0].find(qn('rPr')) if runs else None
            for ch in list(p):
                if ch.tag != qn('pPr'):
                    p.remove(ch)
            r = etree.SubElement(p, qn('r'))
            if rpr is not None:
                r.append(copy.deepcopy(rpr))
            t = etree.SubElement(r, qn('t'))
            t.text = txt
    return new

def append_sentence(p, text):
    """在段落末尾追加一句：深拷贝段落最后一个 run 以继承字体格式。"""
    runs = p.findall(qn('r'))
    assert runs, 'paragraph has no run to clone'
    newr = copy.deepcopy(runs[-1])
    strip_ids(newr)
    for ch in list(newr):
        if ch.tag != qn('rPr'):
            newr.remove(ch)
    t = etree.SubElement(newr, qn('t'))
    t.text = text
    p.append(newr)

# ---------- 定位锚点（任何改动前一次性抓取元素引用） ----------
A = {}
A['abs']       = get_anchor('abstract',   lambda s: s.startswith('摘要：'))
A['intro_end'] = get_anchor('intro last', lambda s: s.startswith('面向算力受限'))
A['h2']        = get_anchor('h 2',        lambda s: s == '2 相关工作')
A['p23']       = get_anchor('2.3 body',   lambda s: s.startswith('时序检测可通过'))
A['h3']        = get_anchor('h 3',        lambda s: s == '3 检测任务与评测协议')
A['p3_first']  = get_anchor('3 body',     lambda s: s.startswith('本文将CST组织为'))
A['h33']       = get_anchor('h 3.3',      lambda s: s == '3.3 精度、误报与及时检出')
A['h4']        = get_anchor('h 4',        lambda s: s == '4 方法与实验结果')
A['src_body']  = get_anchor('body tmpl',  lambda s: s.startswith('τ=0.25时，总体DR'))
A['h5']        = get_anchor('h 5',        lambda s: s == '5 分析与讨论')
A['h51']       = get_anchor('h 5.1',      lambda s: s == '5.1 属性检出与背景误报')
A['h52']       = get_anchor('h 5.2',      lambda s: s == '5.2 重新出现后的及时检出')
A['h53']       = get_anchor('h 5.3',      lambda s: s == '5.3 消融、开销与适用范围')
A['cap1']      = get_anchor('cap 表1',    lambda s: s.startswith('表1　'))
A['h6']        = get_anchor('h 6',        lambda s: s == '6 结论')
A['concl']     = get_anchor('conclusion', lambda s: s.startswith('本文基于'))

tbl1 = d.tables[0]._tbl
assert para_text(tbl1.findall(qn('tr'))[0].findall(qn('tc'))[0]) == '配置', 'table[0] is not 表1'

# ---------- 文本常量（逐字用；引号按原文档使用弯引号“”） ----------
TCE_BLOCK = ('在此基础上，本文提出时序类别编码与解耦监督（Temporal Class Encoding with '
             'Decoupled Supervision, TCE）：利用数据集原生逐帧标注，将堆叠输入中历史通道的残影'
             '转写为远/近/当前三类时序标签，推理时仅解码当前类；同时采用实例级分类—回归解耦监督，'
             '使残影仅参与分类损失、不占用共享回归头的回归容量。TCE不改动网络结构，使mAP50提升2.23'
             '个百分点、帧级检出率提升1.28个百分点，mAP50-95保持不变，背景误报相对下降31.8%。')
CLOSING = ('这种“协议诊断—监督设计—协议验证”的闭环不依赖网络结构改动，可无损移植到任何基于'
           '堆叠输入的检测流程，为评测基准驱动的方法设计提供了范式。')
E2 = ('在上述协议与时序输入研究的基础上，本文进一步提出时序类别编码与解耦监督（TCE）作为方法层面'
      '的改进：其一，利用CST原生逐帧标注，将堆叠输入中历史通道的残影转写为显式时序类别监督；其二，'
      '推理仅解码当前类，使残影印记在部署输出中不再有出口；其三，针对共享回归头的梯度稀释效应，'
      '采用实例级分类—回归解耦监督，残影仅参与分类损失。TCE不改动网络结构，在三种子评测下使mAP50'
      '提升2.23个百分点、帧级检出率提升1.28个百分点，mAP50-95保持不变，背景误报相对下降31.8%。')
E3 = ('上述方法均以输入或特征为时序信息的载体，未利用数据集已有的逐帧标注结构；本文的TCE将逐帧'
      '原生标注转写为时序类别监督，并以解耦监督规避回归容量稀释，属于监督设计层面的时序利用，'
      '与现有方向互补。')
E4_LEAD = ('本节首先将CST组织为单类无人机检测的评测协议（3.1—3.3）；随后针对协议诊断出的残影'
           '误报权衡，在3.4节提出时序类别编码与解耦监督方法。协议为方法提供问题定位与统一验证口径，'
           '方法反过来检验协议的诊断与区分能力。')
S34 = [
 ('三帧堆叠输入显著提升了弱小目标的检出精度，同时引入了系统性的残影误报：在无目标帧上，历史通道'
  '中残留的过去时刻目标印记仍会被检测器当作当前目标输出。误报构成分析显示，视外状态贡献了三帧模型'
  '全部误报帧的63.65%–70.28%，是残影误报的主要来源。其根源在于监督信号的缺口：训练标签沿用数据集'
  '原生标注，仅在当前帧存在目标时提供正样本框，堆叠输入中历史通道的残影从未被标注，训练集中86.3%'
  '的堆叠图平均携带1.97个未标注残影。模型从正样本中习得的先验是“存在目标印记即输出检出”，且不区分'
  '印记所属时刻；当无目标帧上仅剩残影时，模型未被明确告知该印记属于过去而非当前，残影误报由此难以避免。'),
 ('针对上述监督缺口，本文提出时序类别编码与解耦监督（Temporal Class Encoding with Decoupled '
  'Supervision, TCE）。该方法仅作用于监督设计层，不改动网络结构与推理流程，由三个组件构成。其一，'
  '残影标签补全：利用数据集原生逐帧标注，对每幅训练与验证图像，沿其所在序列向前查找第10与第5采样帧'
  '（远端与近端）的原始标注框，并将框坐标复制为该图的残影正样本；历史帧缺失时向当前帧方向钳制，'
  '残影框数量由数据自然决定。其二，三类时序编码：将标签类别数nc由1扩展为3，远端残影、近端残影与'
  '当前帧目标分别编码为far、near与current三类，使模型在训练中学习区分目标印记所属时刻。其三，'
  '当前类解码：推理时仅保留current类的输出作为检测结果，far与near类输出全部丢弃，残影印记在部署'
  '输出中不再有出口，且检测输出与单类基线在同一口径下可比。'),
 ('若将残影框作为普通正样本直接纳入全部监督，会损害当前类的定位精度。YOLO系检测器的框回归与分布'
  '回归（box/DFL）分支为所有类别共享，而残影实例约占全部正样本的2/3；若残影全面参与回归，共享回归头'
  '相当比例的梯度预算将用于推理时被丢弃的残影类输出，稀释当前类在高IoU阈值下的定位精度。为此，TCE'
  '采用实例级分类—回归解耦监督：残影实例仅参与分类损失，其对应锚点上的回归与分布损失权重置零；'
  '同时令回归损失的归一化对齐单类基线，使每个当前类锚点获得的回归梯度尺度与单类训练严格一致。'
  '其设计原则可概括为：残影的存在性与时序属性是可靠信号，其位置是冗余信息，不应占用共享回归头的'
  '回归容量（机理消融见4.5节）。'),
 ('需要说明的是，该解耦与YOLOX等检测器在结构上分离分类与回归分支的解耦头设计不同：后者属于网络'
  '结构层面的解耦，本文则不改动任何网络结构，仅修改损失权重与归一化方式，解耦完全发生在监督信号'
  '层面，因而可无损移植到任何共享回归头的检测框架。'),
 ('残影标签的生成完全自动：仅需读取数据集自带的逐帧标注并按序列名与原帧索引对齐，无需人工标注或'
  '额外信息；测试集标签保持数据集原生标注不变，评测时模型仅输出当前类，与单类基线采用完全相同的'
  '评价口径，保证了方法改进前后结果的可比性与评测公平。'),
]
P45_1 = ('表4给出TCE与三帧单类基线（T1）在三种子下的对照。mAP50提升2.23个百分点'
         '（74.22±1.48→76.45±1.61），帧级检出率DR提升1.28个百分点且种子间标准差由3.29收窄至0.63；'
         'mAP50-95保持不变（39.60±0.59→39.61±1.11）；BG-FP@0.25由15.41±4.77降至10.51±2.01，'
         '相对下降31.8%。以固定τ=0.25同口径计算，R₁为36.89±1.97（基线36.14±2.34），R₃为54.87±0.86'
         '（基线57.49±4.06），重新出现窗口内的及时检出与基线相当，工作点更稳定。')
P45_2 = ('表5将机理拆解为消融。TCE-full（三类编码+残影全回归）的mAP50-95由39.26降至36.06，损伤'
         '集中于高IoU段：AP75由36.3降至29.3，AP80由22.5降至15.8，且在慢速与快速目标上均匀分布'
         '（AP75分别下降6.9与6.7个百分点）。该模式排除了残影框与当前框重叠导致的锚点分配竞争'
         '（竞争只会伤害慢速目标），指向共享回归头的梯度稀释：残影实例约占全部正样本的2/3，共享'
         'box/DFL回归头的梯度预算约73%用于推理时被丢弃的残影类输出。解耦监督将残影限制在分类通路并使'
         '回归归一化对齐单类基线后，mAP50-95恢复至38.98，同时BG-FP由13.93进一步降至8.95，验证了'
         '“残影的存在性与时序属性是信号、其位置不应占用回归容量”的设计原则。')
CAP4 = '表4　TCE与三帧基线对照（统一图像组YOLO11n，test集，τ=0.25，均值±样本标准差，3种子）。'
CAP5 = '表5　机理消融（seed0）。TCE-full为三类编码+残影全回归对照；AP75为固定预测集上的离线重算值。'
T4 = [
 ['输入与方法', 'mAP50 (%)', 'mAP50-95 (%)', 'DR@0.25 (%)', 'BG-FP@0.25 (%)'],
 ['三帧单类基线（T1）', '74.22±1.48', '39.60±0.59', '74.10±3.29', '15.41±4.77'],
 ['三帧+TCE（本文）', '76.45±1.61', '39.61±1.11', '75.38±0.63', '10.51±2.01'],
]
T5 = [
 ['配置', 'mAP50 (%)', 'mAP50-95 (%)', 'AP75 (%)', 'BG-FP@0.25 (%)'],
 ['T1 三帧基线（nc=1）', '74.73', '39.26', '36.3', '20.91'],
 ['TCE-full（nc=3，全回归）', '75.29', '36.06', '29.3', '13.93'],
 ['TCE（nc=3，解耦监督）', '75.22', '38.98', '33.1', '8.95'],
]
E7_A = ('在方法层面，本文提出时序类别编码与解耦监督：将堆叠输入中的残影转写为显式时序类别监督并仅'
        '解码当前类，以实例级解耦监督规避共享回归头的容量稀释，在mAP50-95不受损的前提下同时取得'
        '检出提升（mAP50 +2.23、DR +1.28个百分点）与背景误报下降（相对31.8%）。')
E7_B = ('该改进不改动网络结构，可无损移植到任何基于堆叠输入的检测流程；“协议诊断—监督设计—协议'
        '验证”的闭环也体现了评测基准在反无人机方法设计中的应用前景。')

# ---------- E1 重写摘要 ----------
orig = para_text(A['abs'])
prefix_end = '结果表明，三帧时序输入以较低的附加开销增强了CST场景中弱小无人机的检出能力，并改善了目标重新出现后的及时响应。'
idx = orig.find(prefix_end)
assert idx >= 0, 'abstract prefix sentence not found'
prefix = orig[:idx + len(prefix_end)]
assert '同时，这种时序信息的引入也伴随着部分复杂背景子集误报上升的权衡。' in orig[idx:], \
    'expected original ending sentence right after prefix'
new_abs = prefix + TCE_BLOCK + CLOSING
set_text(A['abs'], new_abs)

# ---------- E2 引言末尾追加一段 ----------
A['h2'].addprevious(clone_para(A['src_body'], E2))

# ---------- E3 2.3 节末尾追加一句 ----------
append_sentence(A['p23'], E3)

# ---------- E4 第3节标题与总起 ----------
set_text(A['h3'], '3 检测任务、评测协议与时序监督方法')
A['p3_first'].addprevious(clone_para(A['src_body'], E4_LEAD))

# ---------- E5 新增 3.4 小节（插在 4 标题之前） ----------
h34 = clone_para(A['h33'], '3.4 时序类别编码与解耦监督（TCE）')
A['h4'].addprevious(h34)
for txt in S34:
    A['h4'].addprevious(clone_para(A['src_body'], txt))

# ---------- E6 第4、5节重组 ----------
set_text(A['h4'], '4 实验结果与分析')
set_text(A['h51'], '4.3 属性检出与背景误报')
set_text(A['h52'], '4.4 重新出现后的及时检出')
set_text(A['h53'], '4.6 消融、开销与适用范围')
A['h5'].getparent().remove(A['h5'])  # 删除“5 分析与讨论”标题
h45 = clone_para(A['h33'], '4.5 时序类别编码的主结果与机理消融')
cap4 = clone_para(A['cap1'], CAP4)
tbl4 = make_table(tbl1, T4)
cap5 = clone_para(A['cap1'], CAP5)
tbl5 = make_table(tbl1, T5)
for el in (h45, clone_para(A['src_body'], P45_1), cap4, tbl4,
           clone_para(A['src_body'], P45_2), cap5, tbl5):
    A['h53'].addprevious(el)

# ---------- E7 结论重组 ----------
set_text(A['h6'], '5 总结')
pA = clone_para(A['src_body'], E7_A)
pB = clone_para(A['src_body'], E7_B)
A['concl'].addnext(pA)
pA.addnext(pB)

d.save(PATH)
print('OK: all edits applied and saved to', PATH)
