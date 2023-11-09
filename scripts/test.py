import os 
print(os.cpu_count())
import time
while True:
    time.sleep(10)
    print(len(os.listdir('/home/feiyuehchen/personality/music_dataset/msd_sheet/mid')))