from __future__ import absolute_import, print_function, unicode_literals

from pyreadline3.py3k_compat import is_ironpython

if is_ironpython:
    try:
        from .ironpython_clipboard import GetClipboardText, SetClipboardText
    except ImportError:
        from .no_clipboard import GetClipboardText, SetClipboardText

else:
    try:
        from .win32_clipboard import GetClipboardText, SetClipboardText
    except ImportError:
        from .no_clipboard import GetClipboardText, SetClipboardText


def send_data(lists):
    SetClipboardText(make_tab(lists))


def set_clipboard_text(toclipboard):
    SetClipboardText(str(toclipboard))


def make_tab(lists):
    if hasattr(lists, "tolist"):
        lists = lists.tolist()
    ut = []
    for rad in lists:
        if type(rad) in [list, tuple]:
            ut.append("\t".join(["%s" % x for x in rad]))
        else:
            ut.append("%s" % rad)
    return "\n".join(ut)


def make_list_of_list(txt):
    def make_num(x):
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                try:
                    return complex(x)
                except ValueError:
                    return x
        return x
    ut = []
    flag = False
    for rad in [x for x in txt.split("\r\n") if x != ""]:
        raden = [make_num(x) for x in rad.split("\t")]
        if str in list(map(type, raden)):
            flag = True
        ut.append(raden)
    return ut, flag


def get_clipboard_text_and_convert(paste_list=False):
    """Get txt from clipboard. if paste_list==True the convert tab separated
    data to list of lists. Enclose list of list in array() if all elements are
    numeric"""
    txt = GetClipboardText()
    if txt:
        if paste_list and "\t" in txt:
            array, flag = make_list_of_list(txt)
            if flag:
                txt = repr(array)
            else:
                txt = "array(%s)" % repr(array)
            txt = "".join([c for c in txt if c not in " \t\r\n"])
    return txt
