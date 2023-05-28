import cv2
from tqdm import tqdm
from os import path

import os
import shutil

import numpy as np

  # Define Path
basePath = os.getcwd()
cmd = f"cd {basePath}"
os.system(cmd)

wav2lipFolderName = "Wav2Lip"
gfpganFolderName = "GFPGAN"
wav2lipPath = f"{basePath}/{wav2lipFolderName}"
gfpganPath = f"{basePath}/{gfpganFolderName}"

inputPath = f"{basePath}/inputs"
inputVideoPath = f"{inputPath}/image"
inputAudioPath = f"{inputPath}/voice"

outputPath = f"{basePath}/outputs"
wav2lipOutputPath = f"{outputPath}/{wav2lipFolderName}"
gfpganBaseOutputPath = f"{outputPath}/{gfpganFolderName}"

if os.path.exists(outputPath):
  shutil.rmtree(outputPath)
  os.makedirs(outputPath)

if not os.path.exists(wav2lipOutputPath):
  os.makedirs(wav2lipOutputPath)
if not os.path.exists(gfpganBaseOutputPath):
  os.makedirs(gfpganBaseOutputPath)

inputVideoFiles = os.listdir(inputVideoPath)
inputAudioFiles = os.listdir(inputAudioPath)

inputVideoFiles.sort()
inputAudioFiles.sort()

if len(inputVideoFiles) > 0:
  for inputVideoFile in inputVideoFiles:
     inputVideoFileName, inputVideoFileExt = os.path.splitext(inputVideoFile)
     if ".ds_store" in inputVideoFile.lower():
        continue
     
     if (inputVideoFileExt == ".bmp" or
        inputVideoFileExt == ".png" or
        inputVideoFileExt == ".jpg" or
        inputVideoFileExt == ".jpeg"):
        inputVideoFileExt=".mp4"

     if len(inputAudioFiles) > 0:
        IsMatch = False
        for inputAudioFile in inputAudioFiles:
          inputAudioFileName, inputAudioFileExt = os.path.splitext(inputAudioFile)
          if ".ds_store" in inputAudioFile.lower():
              continue
          
          if inputVideoFileName == inputAudioFileName:
              IsMatch = True
              break
          
        if IsMatch == True:
           InputVideoFilePath = os.path.join(inputVideoPath, inputVideoFile)
           InputAudioFilePath = os.path.join(inputAudioPath, inputAudioFile)

           Wav2LipOutputFileName = f"Wav2Lip_{inputVideoFileName}{inputVideoFileExt}"
           Wav2LipOutputFilePath = os.path.join(wav2lipOutputPath, Wav2LipOutputFileName)
           cmd = f"cd {wav2lipPath} && python3 inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --fps 60 --pads 0 20 0 0 --face {InputVideoFilePath} --audio {InputAudioFilePath} --outfile {Wav2LipOutputFilePath}"
           os.system(cmd)
           print(f"Wav2Lip End : {Wav2LipOutputFileName}")

           gfpganOutputPath = f"{gfpganBaseOutputPath}/{inputVideoFileName}"
           unProcessedFramesFolderPath = f"{gfpganOutputPath}/org_imgs"
           restoredFramesFolderPath = f"{gfpganOutputPath}/restored_imgs"

           if not os.path.exists(gfpganOutputPath):
             os.makedirs(gfpganOutputPath)
           if not os.path.exists(unProcessedFramesFolderPath):
             os.makedirs(unProcessedFramesFolderPath)

           vidcap = cv2.VideoCapture(Wav2LipOutputFilePath)
           numberOfFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
           fps = vidcap.get(cv2.CAP_PROP_FPS)
           print("FPS: ", fps, "Frames: ", numberOfFrames)

           for frameNumber in tqdm(range(numberOfFrames)):
              _,image = vidcap.read()
              if image is None:
                continue
              cv2.imwrite(path.join(unProcessedFramesFolderPath, str(frameNumber).zfill(4)+'.jpg'), image)
            
           cmd = f"cd {gfpganPath} && python3 inference_gfpgan.py -i {unProcessedFramesFolderPath} -o {gfpganOutputPath} -v 1.3 -s 1 --only_center_face --bg_upsampler None"
           os.system(cmd)
           print(f"GFPGAN End : {inputVideoFile}")

           restoredFrames = os.listdir(restoredFramesFolderPath)
           restoredFrames.sort()

           batch = 0
           batchSize = numberOfFrames
           from tqdm import tqdm
           for i in tqdm(range(0, len(restoredFrames), batchSize)):
              img_array = []
              start, end = i, i+batchSize
              print("processing ", start, end)
              for restoredFrameName in  tqdm(restoredFrames[start:end]):
                restoredFrameName = f"{restoredFramesFolderPath}/{restoredFrameName}";
                img = cv2.imread(restoredFrameName)
                if img is None:
                  continue
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)

              out = cv2.VideoWriter(f"{gfpganOutputPath}/restored_{str(batch).zfill(4)}{inputVideoFileExt}",cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
              batch = batch + 1

              for i in range(len(img_array)):
                out.write(img_array[i])
              out.release()

           concatTextFilePath = gfpganOutputPath + "/concat.txt"
           concatTextFile=open(concatTextFilePath,"w")
           for ips in range(batch):
              concatTextFile.write(f"file restored_{str(ips).zfill(4)}{inputVideoFileExt}\n")
           concatTextFile.close()

           concatedVideoOutputPath = f"{gfpganOutputPath}/concated_output{inputVideoFileExt}"
           cmd = f"ffmpeg -y -f concat -i {concatTextFilePath} -c copy {concatedVideoOutputPath}"
           os.system(cmd)

           outputVideoFile = f"{inputVideoFileName}{inputVideoFileExt}"
           finalProcessedOuputVideo = f"{outputPath}/{outputVideoFile}"
           cmd = f"ffmpeg -y -i {concatedVideoOutputPath} -i {InputAudioFilePath} -map 0 -map 1:a -c:v copy -shortest {finalProcessedOuputVideo}"
           os.system(cmd)
           print("Wav2Lip + GFPGAN End.")
        else:
           continue

     else:
        break
     
