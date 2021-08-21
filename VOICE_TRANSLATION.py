from gtts import gTTS

import googletrans

from playsound import playsound

import os

import time

try:
    print('Google翻譯API 支援以下語系的互轉翻譯:\n',googletrans.LANGUAGES)
    print('\n')
    
    while(True):
        translator = googletrans.Translator()

        file_flag = 0        
        while (file_flag == 0):
            file_name = input('請輸入語音檔案名稱(英文): ')
            if (file_name == ''):
                print('未輸入檔名，則以時間數值進行檔案命名。')
                file_name = str(int(time.time()))                
                file_path = file_name + '.mp3'
            else:
                file_path = file_name + '.mp3'


            if (os.path.isfile(file_path) != True):
                file_flag = 1
                print('\n建立新檔案: ' + file_path)               
            else:
                file_flag = 0
                print('\n檔案已存在，請更換檔案名稱。')

        text_flag = 0
        while (text_flag == 0):
            text = input('請輸入文字訊息內容 (中文 / 英文 皆可): ')
            text_language = translator.detect(text)
            print('原始語言為: ' + text_language.lang + '\n')
            if ((text_language.lang == 'zh-tw')
                or (text_language.lang == 'zh-CN')
                or (text_language.lang == 'en')):
                text_flag = 1

        if (text_language.lang == 'zh-tw') or (text_language.lang == 'zh-CN'):
            text_2 = translator.translate(text, dest='en').text
            print('原始語言為中文，翻譯目的語言為英文。')
            language = 'zh'
            language_2 = 'en'

        if (text_language.lang == 'en') :
            text_2 = translator.translate(text, dest='zh-tw').text
            print('原始語言為英文，翻譯目的語言為中文。')
            language = 'en'
            language_2 = 'zh'

        print('\n翻譯結果:\n',text_2 + '\n')

        speech = gTTS(text = text, lang = language, slow = False)
        speech.save(file_name + '.mp3')
        playsound(file_name + '.mp3')

        speech = gTTS(text = text_2, lang = language_2, slow = False)
        speech.save(file_name + '_2.mp3')
        playsound(file_name + '_2.mp3')

except KeyboardInterrupt:
    print(" Quit")
