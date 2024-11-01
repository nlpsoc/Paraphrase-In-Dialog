"""
    test tex highlighting
"""
from unittest import TestCase
from paraphrase.utility.tex_highlighting import generate, _gen_latex_text_hl, gen_guest_host_tex


class Test(TestCase):
    def test_generate(self):
        tex_str_guest = _gen_latex_text_hl(["This", "is", "the", "guest", "message."], [0, 5, 4, 9, 10], hl_color="red",
                                           rescale_value=True)
        result_str = r'''\begin{CJK*}{UTF8}{gbsn}''' + '\n' \
                     + r'''{\setlength{\fboxsep}{0pt}\colorbox{white!0}{\parbox{0.9\textwidth}{''' + '\n' \
                     + r'''\colorbox{red!0.0}{\strut This} \colorbox{red!50.0}{\strut is} \colorbox{red!40.0}{\strut the} \colorbox{red!90.0}{\strut guest} \colorbox{red!100.0}{\strut message.} ''' \
                     + '\n' + r'''}}}''' + '\n' + r'''\end{CJK*}'''
        self.assertEqual(tex_str_guest, result_str)
        tex_str_host = _gen_latex_text_hl(["This", "is", "the", "host", "reply."], [0, 5, 1, 6, 10], hl_color="blue",
                                          rescale_value=True)

        combi_str = gen_guest_host_tex(
            [["This", "is", "the", "guest", "message."], ["This", "is", "the", "host", "reply."]],
            [[0, 5, 4, 9, 10], [0, 5, 1, 6, 10]])
        print(combi_str)
