from gtts import gTTS

from playsound import playsound

import os

import time

try:
    playsound('welcome_2.mp3')

    while(True):
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
                playsound('hint.mp3')


        text = input('請輸入語音訊息內容: ')
        language = input('請輸入語系代碼: ')

        speech = gTTS(text = text, lang = language, slow = False)

        speech.save(file_name + '.mp3')

        playsound(file_name + '.mp3')

        print('\n')
        
except KeyboardInterrupt:
    print(" Quit")
    playsound('bye.mp3')
