import os
import glob

# s2 and s6 for unseen speakers
def do_split(root):
    ftra = open('unseen_train.csv', 'w')
    ftst = open('unseen_test.csv', 'w')
    for spk in os.listdir(root):  # s1
        seg_dir = os.path.join(root, spk)
        for seg in os.listdir(seg_dir):  # s1/20090702
            wav_dir = os.path.join(seg_dir, seg)
            for wav in os.listdir(wav_dir):  # s1/20090702/xx.wav
                wavname = os.path.splitext(wav)[0]
                valid_path = spk+'/'+seg+'_'+wavname
                if spk in ['s2', 's6']:
                    ftst.write(valid_path+'\n')
                else:
                    ftra.write(valid_path+'\n')
    ftra.close()
    ftst.close()
    print('Done')



do_split(r'D:\LipData\CMLR\audio')
        

