import os

path = "/root/userfolder/hf2vad/data/shanghaitech/training/videos"

pathlist = os.listdir(path)

pathname = []

for i in pathlist:
    pathname.append(i.split(".")[0])

for filename in pathname:
    #ffmpeg - i training/videos/01_001.avi training/frames/01_001/%04d.jpg
    # cmd = 'mkdir training/frames/'+filename
    # os.system(cmd)
    cmd = 'ffmpeg -i /root/userfolder/hf2vad/data/shanghaitech/training/videos/'+filename+'.avi -qscale:v 1 -qmin 1 /root/userfolder/hf2vad/data/shanghaitech/training/frames/'+filename+'/%04d.jpg'
    os.system(cmd)
