# _*_ coding: utf-8 _*_

# import subprocess
# from subprocess import PIPE

# program = 'python observation.py'
# args = 'O3'
#
# process = subprocess.Popen(program + ' ' + args, shell=True, stdout=subprocess.PIPE)
#
# value, err = process.communicate()  # 主要用來取回 stdout 跟 stderr 的輸出
#
# print('---')
# print(value.split()[-1])

# -----

from gtts import gTTS
import pygame

string = '好魚哥   土雞王'

tts = gTTS(text=string, lang='zh-tw')
tts.save('voice.mp3')

pygame.mixer.init()
pygame.mixer.music.load('voice.mp3')
pygame.mixer.music.play()

while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
