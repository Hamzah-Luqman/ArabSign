from featuresExtractor import features



##
diVideoSet = {"dataset" : "ArabSign",
    "nClasses" : 50,   # number of classes
    "nFramesNorm" : 80,    # number of frames per video
    "nMinDim" : 224,   # smaller dimension of saved video-frames
    "tuShape" : (224, 224), # height, width
    "nFpsAvg" : 30,
    "nFramesAvg" : 50, 
    "fDurationAvg" : 2.0,# seconds 
    "reshape_input": False}  #True: if the raw input is different from the requested shape for the model

# feature extractor 
diFeature = {"model" : "mobilenet",
    "tuInputShape" : (224, 224, 3),
    "tuOutputShape" : (1024, )} 

# Video and Frames paths
extractFrames = False
FramesPath  = ''

# Features destination path
extractFeatures = False
destFeaturesPath = ''

# Extract features from video frames
if extractFeatures:
    extractor = features(FramesPath, destFeaturesPath, diFeature)
    print('============== START OF FEATURES EXTRACTION ====================')
    extractor.extractFeatures()
    print('============== END OF FEATURES EXTRACTION ====================')

# Model train and test



