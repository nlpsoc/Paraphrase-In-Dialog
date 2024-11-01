"""
    generate tex code for highlights

    original copied from https://raw.githubusercontent.com/jiesutd/Text-Attention-Heatmap-Visualization/master/text_attention.py
"""

## convert the text/attention list to latex code, which will further generates the text heatmap based on attention weights.
import numpy as np

latex_special_token = ["!@#$%^&*()"]


def generate(text_list, attention_list, latex_file, hl_color='red', rescale_value=False):
    with open(latex_file, 'w') as f:
        f.write(r'''\documentclass[varwidth]{standalone}
                \special{papersize=210mm,297mm}
                \usepackage{color}
                \usepackage{tcolorbox}
                \usepackage{CJK}
                \usepackage{adjustbox}
                \tcbset{width=0.9\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
                \begin{document}''')

        string = _gen_latex_text_hl(text_list, attention_list, hl_color, rescale_value)

        f.write(string)
        f.write(r'''\end{document}''')


def _gen_latex_text_hl(text_list, attention_list, hl_color='purple', rescale_value=False, speaker="Guest",
                       table_format=True):
    assert (len(text_list) == len(attention_list))
    if rescale_value:
        attention_list = rescale(attention_list)
        attention_list = [0 if np.isnan(weight) else weight for weight in attention_list]
    text_list = clean_word(text_list)
    if table_format:
        tex_str = _create_latex_hl_string(attention_list, hl_color, speaker, text_list)
    else:
        tex_str = _create_latex_CJK_string(attention_list, hl_color, speaker, text_list)

    return tex_str


def _create_latex_CJK_string(attention_list, hl_color, speaker, text_list):
    nbr_words = len(text_list)
    tex_str = r'''\begin{CJK*}{UTF8}{gbsn}''' + '\n'
    tex_str += r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{\textbf{''' + speaker + ": }\n"
    for idx in range(nbr_words):
        tex_str += "\\colorbox{%s!%s}{" % (hl_color, attention_list[idx]) + "\\strut " + text_list[idx] + "} "
    tex_str += "\n}}}"
    tex_str += '\n' + r'''\end{CJK*}'''
    return tex_str


def _create_latex_hl_string(attention_list, hl_color, speaker, text_list):
    nbr_words = len(text_list)
    # string = r'''\begin{CJK*}{UTF8}{gbsn}''' + '\n'
    tex_str = r'''\textbf{''' + speaker + ": }\n"
    idx_l = 0
    idx_r = 1
    while idx_l < nbr_words:

        #  GET the index of the next word with a different highlight value
        while (idx_r < nbr_words) and (attention_list[idx_l] == attention_list[idx_r]):
            idx_r += 1

        color_name = hl_color + str(int(attention_list[idx_l]))
        tex_str += r'''\colorlet{''' + color_name + r'''}{''' + hl_color + '!' + str(attention_list[idx_l]) + r'''}'''
        tex_str += r'''\sethlcolor{''' + color_name + r'''}\hl{'''
        for cur_id in range(idx_l, idx_r):
            tex_str += text_list[cur_id]
            tex_str += " "
        tex_str += "} "
        idx_l = idx_r

    return tex_str


def gen_guest_host_tex(text_lists, attention_lists):
    assert (len(text_lists) == 2)
    assert (len(attention_lists) == 2)
    guests_str = _gen_latex_text_hl(text_lists[0], [elem * 100 for elem in attention_lists[0]], hl_color='purple',
                                    speaker="Guest")
    host_str = _gen_latex_text_hl(text_lists[1], [elem * 100 for elem in attention_lists[1]], hl_color='orange',
                                  speaker="Host")
    return guests_str + "\n \\\\ \n" + host_str


def rescale(input_list):
    the_array = np.asarray(input_list)
    the_max = np.max(the_array)
    the_min = np.min(the_array)
    rescale = (the_array - the_min) / (the_max - the_min) * 100
    return rescale.tolist()


def clean_word(word_list):
    new_word_list = []
    for word in word_list:
        for latex_sensitive in ["\\", "%", "&", "^", "#", "_", "{", "}"]:
            if latex_sensitive in word:
                word = word.replace(latex_sensitive, '\\' + latex_sensitive)
        new_word_list.append(word)
    return new_word_list


if __name__ == '__main__':
    ## This is a demo:

    sent = '''the USS Ronald Reagan - an aircraft carrier docked in Japan - during his tour of the region, vowing to "defeat any attack and meet any use of conventional or nuclear weapons with an overwhelming and effective American response".
North Korea and the US have ratcheted up tensions in recent weeks and the movement of the strike group had raised the question of a pre-emptive strike by the US.
On Wednesday, Mr Pence described the country as the "most dangerous and urgent threat to peace and security" in the Asia-Pacific.'''
    sent = '''我 回忆 起 我 曾经 在 大学 年代 ， 我们 经常 喜欢 玩 “ Hawaii guitar ” 。 说起 Guitar ， 我 想起 了 西游记 里 的 琵琶精 。
	今年 下半年 ， 中 美 合拍 的 西游记 即将 正式 开机 ， 我 继续 扮演 美猴王 孙悟空 ， 我 会 用 美猴王 艺术 形象 努力 创造 一 个 正能量 的 形象 ， 文 体 两 开花 ， 弘扬 中华 文化 ， 希望 大家 能 多多 关注 。'''
    words = sent.split()
    word_num = len(words)
    attention = [(x + 1.) / word_num * 100 for x in range(word_num)]
    import random

    random.seed(42)
    random.shuffle(attention)
    color = 'red'
    generate(words, attention, "sample.tex", color)
