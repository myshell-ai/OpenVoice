'''
Defines the set of symbols used in text input to the model.
'''

# japanese_cleaners
# _pad        = '_'
# _punctuation = ',.!?-'
# _letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧ↓↑ '


'''# japanese_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'AEINOQUabdefghijkmnoprstuvwyzʃʧʦ↓↑ '
'''


'''# korean_cleaners
_pad        = '_'
_punctuation = ',.!?…~'
_letters = 'ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎㄲㄸㅃㅆㅉㅏㅓㅗㅜㅡㅣㅐㅔ '
'''

'''# chinese_cleaners
_pad        = '_'
_punctuation = '，。！？—…'
_letters = 'ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦㄧㄨㄩˉˊˇˋ˙ '
'''

# # zh_ja_mixture_cleaners
# _pad        = '_'
# _punctuation = ',.!?-~…'
# _letters = 'AEINOQUabdefghijklmnoprstuvwyzʃʧʦɯɹəɥ⁼ʰ`→↓↑ '


'''# sanskrit_cleaners
_pad        = '_'
_punctuation = '।'
_letters = 'ँंःअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलळवशषसहऽािीुूृॄेैोौ्ॠॢ '
'''

'''# cjks_cleaners
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzʃʧʥʦɯɹəɥçɸɾβŋɦː⁼ʰ`^#*=→↓↑ '
'''

'''# thai_cleaners
_pad        = '_'
_punctuation = '.!? '
_letters = 'กขฃคฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์'
'''

# # cjke_cleaners2
_pad        = '_'
_punctuation = ',.!?-~…'
_letters = 'NQabdefghijklmnopstuvwxyzɑæʃʑçɯɪɔɛɹðəɫɥɸʊɾʒθβŋɦ⁼ʰ`^#*=ˈˌ→↓↑ '


'''# shanghainese_cleaners
_pad        = '_'
_punctuation = ',.!?…'
_letters = 'abdfghiklmnopstuvyzøŋȵɑɔɕəɤɦɪɿʑʔʰ̩̃ᴀᴇ15678 '
'''

'''# chinese_dialect_cleaners
_pad        = '_'
_punctuation = ',.!?~…─'
_letters = '#Nabdefghijklmnoprstuvwxyzæçøŋœȵɐɑɒɓɔɕɗɘəɚɛɜɣɤɦɪɭɯɵɷɸɻɾɿʂʅʊʋʌʏʑʔʦʮʰʷˀː˥˦˧˨˩̥̩̃̚ᴀᴇ↑↓∅ⱼ '
'''

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters)

# Special symbol ids
SPACE_ID = symbols.index(" ")

num_ja_tones = 1
num_kr_tones = 1
num_zh_tones = 6
num_en_tones = 4

language_tone_start_map = {
    "ZH": 0,
    "JP": num_zh_tones,
    "EN": num_zh_tones + num_ja_tones,
    'KR': num_zh_tones + num_ja_tones + num_en_tones,
}