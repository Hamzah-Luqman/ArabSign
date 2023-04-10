from models import VGG16FeaturesExtractor
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Extract the features from the video frames (assuming that video frames already extracted)

class features:
    def __init__(self, sourcePath, destPath, diFeature) -> None:
        self.sourcePath = sourcePath
        self.destPath   = destPath
        self.diFeature = diFeature

    def extractFeatures(self):
        if self.diFeature['model'] == 'vgg16':
            model =  VGG16FeaturesExtractor()
        else:
            raise ValueError('Not defined model !')
        
        print(model)

        # The dataset should have the signer\cat[train\test]\class\sample\sampleFrames

        for signer in os.listdir(self.sourcePath):
            signerFolder = os.path.join(self.sourcePath, signer)
            if os.path.isdir(signerFolder):
                signerFolderDest = os.path.join(self.destPath, signer)
                self.createFolder(signerFolderDest)

                for cat in os.listdir(signerFolder):
                    signCat = os.path.join(signerFolder, cat)
                    destCat = os.path.join(signerFolderDest, cat)
                    self.createFolder(destCat)


                    for sign in os.listdir(signCat): 
                        signFolder = os.path.join(signCat, sign)
                        signFolderDest = os.path.join(destCat, sign)
                        self.createFolder(signFolderDest)

                        print(f'Extracting features from {signFolder}')

                        transform = transforms.Compose([
                            transforms.Resize(self.diFeature.tuInputShape[0:2]),
                            transforms.ToTensor()
                        ])

                        dataset = ImageFolder(root=signFolder, transform=transform)
                        loader = DataLoader(dataset, batch_size=80)
                        i = 1
                        print(signFolderDest)
                        for sample, _ in tqdm(loader):
                            filenameDest = signFolderDest+"/"+str(i)+".npy"
                            if os.path.exists(filenameDest) == True:
                                continue

                            features = model(sample) 
                            features_np = features.numpy() 
                            #print(features_np.shape)
                            np.save(filenameDest, features_np)
                            i = i + 1


    def createFolder(self, folderPath):
        if os.path.exists(folderPath) ==  False:
            os.mkdir(folderPath)




