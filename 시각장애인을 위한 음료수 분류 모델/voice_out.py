from webcam_new import cam_recog
from gtts import gTTS
from playsound import playsound


def voice_save(acc, lab):
    f = open('./label.txt', 'w')
    f.write('{}%로 {}입니다'.format(acc, lab))
    f.close()

    f = open('./label.txt', 'r')
    text = f.readline()
    f.close()

    tts = gTTS(text=text, lang='ko')
    tts.save('./tts.mp3')

    playsound('./tts.mp3')


if __name__ == '__main__':
    Accuracy, Label = cam_recog()
    voice_save(Accuracy, Label)
