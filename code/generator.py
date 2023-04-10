# %%
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import torchvision.transforms as transforms
import glob
from PIL import Image as pil
from sklearn.model_selection import train_test_split

# %%
# This cutomDataset to read images per class (not vidoe frames)
class CustomImageDataset(Dataset):
    def __init__(self, dataPath, transformer = None, fileExtention='jpg'):
        self.path = Path(dataPath)
        self.transform = transformer
        self.fileExt = fileExtention
        self.filenames = pd.DataFrame(sorted(glob.glob(dataPath + "/*/*/*."+fileExtention)), columns=["sPath"])
        self.labels = self.filenames.sPath.apply(lambda s: s.split("/")[-3]) 
        self.classes = sorted(list(self.labels.unique()))
        self.nClasses = len(self.classes)
        print(f'Found {len(self.filenames)} images belong to {self.nClasses} classes')
    
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        #print("Index: ", str(index))  
        #print(self.labels)
        
        image_filepath = self.filenames.iloc[index].sPath
        #print(image_filepath)
        image = pil.open(image_filepath)
        label = self.labels.iloc[index]
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image) 
        #vector = torch.from_numpy(np.load(fn))
        return image, label


# %%



# %%
# This cutomDataset to read images per class (not video frames) according to the csv list
class CustomCSVDataset(Dataset):
    def __init__(self, csvDataPath, captionToIndex, captions, transformer = None, cat ='train', split = 0.8, valAvailable = False):
        self.path = csvDataPath
        self.transform = transformer
        #self.fileExt = fileExtention
        self.fileData = pd.read_csv(csvDataPath, dtype = str, names=["index","sentId","sPath","framesNo","signerID", "caption", "procCaption"])

        #print(self.fileData.sPath)
        self.filenames = self.fileData.sPath.apply(self.append_ext)
        self.labels = self.fileData["sentId"] #self.filenames.sPath.apply(lambda s: s.split("/")[-3]) 
        self.classes = sorted(list(self.labels.unique()))
        self.nClasses = len(self.classes)
        print(f'Found {len(self.filenames)} samples belong to {self.nClasses} classes')
        self.captionToIndex = captionToIndex
        self.captions = captions
        #print(self.filenames)
        if valAvailable:
            train_data, val_data, y_train, y_val = train_test_split(self.filenames, self.labels, test_size=split, random_state=42, stratify=self.labels)   
            if cat == 'train':
                self.filenames = train_data
                self.labels = y_train
                self.filenames.reset_index(drop=True, inplace=True)
                print(f'Found {len(self.filenames)} {cat} samples with {len(self.labels)} labels')
            elif cat == 'val':
                self.filenames = val_data
                self.labels = y_val
                self.filenames.reset_index(drop=True, inplace=True)
                print(f'Found {len(self.filenames)} {cat}  samples with {len(self.labels)} labels')
            else:
                print('Data cat is invalid !!!')
            #print(self.filenames)

    def __len__(self):
        return len(self.filenames)
    
            
    def append_ext(self, filePath):
        return filePath + '.npy'
            
    def __getitem__(self, index):
        #print("Index: ", str(index))  
        #print(self.labels)
        
        sample_filepath = self.filenames[index]
        #print(sample_filepath)
        data = np.load(sample_filepath)
        label = self.labels.iloc[index]
        if data.shape[0] != 80:
            print("Less than 80: ", sample_filepath)
        if self.transform is not None:
            image = self.transform(data) 
        #vector = torch.from_numpy(np.load(fn))

        #print(len(self.captions))
        try:
            tokenized_caption = self.captions[int(label) - 1]
        except:
            print(self.captions)
            print(self.captions.shape)
            print('Error : ', int(label))
        
        mapped_caption = []
        # Convert the sentences to their mapping through captionToIndex.
        for tok in tokenized_caption:
            mapped_caption.append(self.captionToIndex[tok])
        #print(mapped_caption)

        return data, torch.tensor(mapped_caption)


'''
# %%
print('hi')
transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                        ])
train_image_paths = "/home/eye/ArSL-Continuous/80/color/01/test"    
train_dataset = CustomImageDataset(train_image_paths,transform)
#print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#batch of image tensor
next(iter(train_loader))[0].shape
'''
'''
# %%
transform = None
train_image_paths = "/home/eye/ArSL-Continuous/80/features/images/vgg/color/01.csv"    
train_dataset = CustomCSVDataset(train_image_paths,transform)
#print('The shape of tensor for 50th image in train dataset: ',train_dataset[49][0].shape)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
#batch of image tensor
next(iter(train_loader))[0].shape

# %%


'''