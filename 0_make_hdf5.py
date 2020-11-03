#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 09:53:14 2020

@author: vicenc
"""

######################################################################################################################################
# Parameters

base_dir = '/home/vicenc/software/Unet_Pytorch/Data'
dataname=''
experiment = ''

#------- Image size parameters

# The images can be of any size or aspect ratio (width, height): We are going to pad them to make them square and to resize before extracting tiles

patch_size_h = 576 #size of the tiles or patches to extract and save in the database, must be >= than training size
patch_size_v = 416 #size of the tiles or patches to extract and save in the database, must be >= than training size

resize_h= patch_size_h+1 #pad and resize input images to (resize x resize) pixels before extracting the tiles.
resize_v= patch_size_v+1

mirror_pad_size=0 # number of pixels to pad *after* resize to image with by mirroring 
step = 1 #distance to skip between patches, 1 indicates pixel wise extraction, patch_size would result in non-overlapping tiles

#-----Note---
#One should likely make sure that  (nrow+mirror_pad_size) mod patch_size == 0, where nrow is the number of rows after resizing
#so that no pixels are lost (any remainer is ignored)

#-------- Training / Validation / Test set parameters

test_set_size=0.1 # what percentage of the dataset should be used as a held out validation/testing set
classes=[0,1] #what classes we expect to have in the data, 


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################


import sklearn
#print(sklearn.__version__)
import tables
import os,sys
import glob
import PIL
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn import model_selection
import sklearn.feature_extraction.image
import sklearn.feature_extraction
import random


######################################################################################################################################

os.chdir(base_dir +'/'+dataname+'/'+experiment)

seed = random.randrange(sys.maxsize) #get a random seed so that we can reproducibly do the cross validation setup
random.seed(seed) # set the seed
#print(f"random seed (note down for reproducibility): {seed}")

img_dtype = tables.UInt8Atom()  # dtype in which the images will be saved, this indicates that images will be saved as unsigned int 8 bit, i.e., [0,255]
filenameAtom = tables.StringAtom(itemsize=255) #create an atom to store the filename of the image, just in case we need it later.

files=glob.glob('../imgs_json_masks/*.png') # create a list of the files, in this case we're only interested in files which have masks so we can use supervised learning

#create training and validation stages and split the files appropriately between them
phases={}
phases["train"],phases["val"]=next(iter(model_selection.ShuffleSplit(n_splits=1,test_size=test_set_size).split(files)))

print(f"\tDataset: {dataname}")
print(f"\tExperiment: {experiment}")
print(f"\tImages size: {patch_size_h} x {patch_size_v}")
print(f"\tNumber of training images: {len(phases['train'])}")
print(f"\tNumber of validation images: {len(phases['val'])}")
print(f"\tTraining / Validation : {100*(1-test_set_size)} % / {100*(test_set_size)} %")


#specify that we'll be saving 2 different image types to the database, an image and its associated masked
imgtypes=["img","mask"]

storage={} # holder for future pytables

block_shape={} #block shape specifies what we'll be saving into the pytable array, here we assume that masks are 1d and images are 3d
block_shape["img"]= np.array((patch_size_v,patch_size_h,3))
block_shape["mask"]= np.array((patch_size_v,patch_size_h)) 

filters=tables.Filters(complevel=6, complib='zlib') #we can also specify filters, such as compression, to improve storage speed


######################################################################################################################################

for phase in phases.keys(): #now for each of the phases (train and validation), we'll loop through the files
    print('')
    print(phase)
    print(f"Storing data in: ./{dataname}_{phase}.pytable")
    
    totals=np.zeros((2,len(classes))) # we counts of all the pixels belonging to (object instance or background) classes
                                        #in for in particular training, since we 
    totals[0,:]=classes               # can later use this information to create better weights for the loss.

    hdf5_file = tables.open_file(f"./{dataname}_{phase}.pytable", mode='w') #open the respective pytable
    storage["filename"] = hdf5_file.create_earray(hdf5_file.root, 'filename', filenameAtom, (0,)) #create the array for storage
    
    for imgtype in imgtypes: #for each of the image types, in this case mask and image, we need to create the associated array
        storage[imgtype]= hdf5_file.create_earray(hdf5_file.root, imgtype, img_dtype,  
                                                  shape=np.append([0],block_shape[imgtype]), 
                                                  chunkshape=np.append([1],block_shape[imgtype]),
                                                  filters=filters)
    
    for filei in phases[phase]: #now for each of the files
        fname=files[filei] 
        print('.', end='')
        for imgtype in imgtypes:
            if(imgtype=="img"): #if we're looking at an img, it must be 3 channel, but cv2 won't load it in the correct channel order, so we need to fix that
                io=cv2.cvtColor(cv2.imread("../imgs_json_masks/"+os.path.basename(fname).replace("_mask.png",".jpg")),cv2.COLOR_BGR2RGB)
                height = io.shape[0]
                width =  io.shape[1]
                '''
                #Convert to square image                
                if height > width:
                    io = cv2.copyMakeBorder(io, 0, 0, 0, height-width, cv2.BORDER_CONSTANT,value=[0,0,0])
                elif height < width :
                    io = cv2.copyMakeBorder( io, 0, width-height, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
                '''
                io = cv2.resize(io,(resize_h,resize_v), interpolation=PIL.Image.LANCZOS) #resize it as specified above
                #plt.imshow(io)

            else: #if its a mask image, then we only need a single channel 
                io=cv2.imread(fname)/255 #the image is loaded as {0,255}, but we'd like to store it as {0,1} since this represents the binary nature of the mask easier
                interp_method=PIL.Image.NEAREST #want to use nearest! otherwise resizing may cause non-existing classes to be produced via interpolation (e.g., ".25")
                height = io.shape[0]
                width =  io.shape[1]
                '''
                #Convert to square image                
                if height > width:
                    io = cv2.copyMakeBorder(io, 0, 0, 0, height-width, cv2.BORDER_CONSTANT,value=[0,0,0])
                elif height < width :
                    io = cv2.copyMakeBorder( io, 0, width-height, 0, 0, cv2.BORDER_CONSTANT,value=[0,0,0])
                '''
                io = cv2.resize(io,(resize_h,resize_v), interpolation=interp_method) #resize it as specified above
                #plt.imshow(io)
                #plt.show()
                
                for i,key in enumerate(classes): #sum the number of pixels, this is done pre-resize, but proportions don't change which is really what we're after
                    totals[1,i]+=sum(sum(io[:,:,0]==key))

            
            io = np.pad(io, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)], mode="reflect")

            #convert input image into overlapping tiles, size is ntiler x ntilec x 1 x patch_size x patch_size x 3          
            #https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/feature_extraction/image.py#L241
            
            io_arr_out=sklearn.feature_extraction.image.extract_patches(io,(patch_size_v,patch_size_h,3),step)
            #resize it into a ntile x patch_size x patch_size x 3
            io_arr_out=io_arr_out.reshape(-1,patch_size_v,patch_size_h,3)
            #plt.imshow(io)
            #plt.show()
            #plt.imshow(io_arr_out[2,:,:,:])
            #plt.show()
            
            #save the 4D tensor to the table
            if(imgtype=="img"):
                storage[imgtype].append(io_arr_out)
            else:
                storage[imgtype].append(io_arr_out[:,:,:,0].squeeze()) #only need 1 channel for mask data

        storage["filename"].append([fname for x in range(io_arr_out.shape[0])]) #add the filename to the storage array
        
    #lastely, we should store the number of pixels
    npixels=hdf5_file.create_carray(hdf5_file.root, 'numpixels', tables.Atom.from_dtype(totals.dtype), totals.shape)
    npixels[:]=totals
    hdf5_file.close()
