# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import re

from pypinyin.contrib.tone_convert import tone_to_tone2, tone2_to_tone

_re_num = re.compile(r'\d')


class ToneSandhiMixin(object):
    """

    按普通话变调规则处理拼音：

    * https://en.wikipedia.org/wiki/Standard_Chinese_phonology#Tone_sandhi
    * https://studycli.org/zh-CN/learn-chinese/tone-changes-in-mandarin/

    """  # noqa

    def post_pinyin(self, han, heteronym, pinyin_list, **kwargs):
        ret = super(ToneSandhiMixin, self).post_pinyin(
            han, heteronym, pinyin_list, **kwargs)
        if ret is not None:
            pinyin_list = ret

        pinyin_list = self._third_tone(han, pinyin_list)
        pinyin_list = self._bu(han, pinyin_list)
        pinyin_list = self._yi(han, pinyin_list)

        return pinyin_list

    def _third_tone(self, han, pinyin_list):
        """

        Third tone sandhi:

        The principal rule of third tone sandhi is:

            When there are two consecutive third-tone syllables, the first of them is pronounced with second tone.

        For example, lǎoshǔ 老鼠 ("mouse") comes to be pronounced láoshǔ [lau̯˧˥ʂu˨˩]. It has been investigated whether the rising contour (˧˥) on the prior syllable is in fact identical to a normal second tone; it has been concluded that it is, at least in terms of auditory perception.[1]: 237 

        When there are three or more third tones in a row, the situation becomes more complicated, since a third tone that precedes a second tone resulting from third tone sandhi may or may not be subject to sandhi itself. The results may depend on word boundaries, stress, and dialectal variations. General rules for three-syllable third-tone combinations can be formulated as follows:

            If the first word is two syllables and the second word is one syllable, then the first two syllables become second tones. For example, bǎoguǎn hǎo 保管好 ("to take good care of") takes the pronunciation báoguán hǎo [pau̯˧˥kwan˧˥xau̯˨˩˦].
            If the first word is one syllable, and the second word is two syllables, the second syllable becomes second tone, but the first syllable remains third tone. For example: lǎo bǎoguǎn 老保管 ("to take care of all the time") takes the pronunciation lǎo báoguǎn [lau̯˨˩pau̯˧˥kwan˨˩˦].

        Some linguists have put forward more comprehensive systems of sandhi rules for multiple third tone sequences. For example, it is proposed[1]: 248  that modifications are applied cyclically, initially within rhythmic feet (trochees; see below), and that sandhi "need not apply between two cyclic branches".

        """  # noqa
        tone2_pinyin_list = [tone_to_tone2(x[0]) for x in pinyin_list]
        if '3' not in ''.join(tone2_pinyin_list):
            return pinyin_list

        changed = False
        third_num = 0
        for pinyin in tone2_pinyin_list:
            if '3' in pinyin:
                third_num += 1
            else:
                third_num = 0

        if third_num == 2:
            for i, v in enumerate(tone2_pinyin_list):
                if '3' in v:
                    tone2_pinyin_list[i] = v.replace('3', '2')
                    changed = True
                    break

        elif third_num > 2:
            n = 1
            for i, v in enumerate(tone2_pinyin_list):
                if '3' in v:
                    if n == third_num:
                        break
                    tone2_pinyin_list[i] = v.replace('3', '2')
                    changed = True
                    n += 1

        if changed:
            return [[tone2_to_tone(x)] for x in tone2_pinyin_list]

        return pinyin_list

    def _bu(self, han, pinyin_list):
        """

        For 不 bù:

            不 is pronounced with second tone when followed by a fourth tone syllable.

                Example: 不是 (bù+shì, "to not be") becomes búshì [pu˧˥ʂɻ̩˥˩]

            In other cases, 不 is pronounced with fourth tone. However, when used between words in an A-not-A question, it may become neutral in tone (e.g., 是不是 shìbushì).

        """  # noqa
        if '不' not in han:
            return pinyin_list

        tone2_pinyin_list = [tone_to_tone2(x[0]) for x in pinyin_list]
        changed = False

        for i, h in enumerate(han):
            current_pinyin = tone2_pinyin_list[i]
            if h == '不' and i < len(han) - 1:
                next_pinyin = tone2_pinyin_list[i+1]
                if '4' in next_pinyin:
                    tone2_pinyin_list[i] = current_pinyin.replace('4', '2')
                    changed = True
                else:
                    tone2_pinyin_list[i] = _re_num.sub('4', current_pinyin)
                    changed = True
            elif h == '不':
                tone2_pinyin_list[i] = _re_num.sub('4', current_pinyin)
                changed = True

        if changed:
            return [[tone2_to_tone(x)] for x in tone2_pinyin_list]

        return pinyin_list

    def _yi(self, han, pinyin_list):
        """

        For 一 yī:

            一 is pronounced with second tone when followed by a fourth tone syllable.

                Example: 一定 (yī+dìng, "must") becomes yídìng [i˧˥tiŋ˥˩]

            Before a first, second or third tone syllable, 一 is pronounced with fourth tone.

                Examples：一天 (yī+tiān, "one day") becomes yìtiān [i˥˩tʰjɛn˥], 一年 (yī+nián, "one year") becomes yìnián [i˥˩njɛn˧˥], 一起 (yī+qǐ, "together") becomes yìqǐ [i˥˩t͡ɕʰi˨˩˦].

            When final, or when it comes at the end of a multi-syllable word (regardless of the first tone of the next word), 一 is pronounced with first tone. It also has first tone when used as an ordinal number (or part of one), and when it is immediately followed by any digit (including another 一; hence both syllables of the word 一一 yīyī and its compounds have first tone).
            When 一 is used between two reduplicated words, it may become neutral in tone (e.g. 看一看 kànyikàn ("to take a look of")).

        """  # noqa
        if '一' not in han:
            return pinyin_list

        tone2_pinyin_list = [tone_to_tone2(x[0]) for x in pinyin_list]
        changed = False

        for i, h in enumerate(han):
            current_pinyin = tone2_pinyin_list[i]
            if h == '一' and i < len(han) - 1:
                next_pinyin = tone2_pinyin_list[i + 1]
                if '4' in next_pinyin:
                    tone2_pinyin_list[i] = current_pinyin.replace('4', '2')
                    changed = True
                else:
                    tone2_pinyin_list[i] = _re_num.sub('4', current_pinyin)
                    changed = True
            elif h == '一':
                tone2_pinyin_list[i] = _re_num.sub('1', current_pinyin)
                changed = True

        if changed:
            return [[tone2_to_tone(x)] for x in tone2_pinyin_list]

        return pinyin_list
