import torch.nn as nn

def explainModel(model, Hin, Win, desiredHout, desiredWout):
    #Hin = input.shape[0]
    #Win = input.shape[1]
    print("Information about model "+model.__class__.__name__)
    layerHin = Hin
    layerWin = Win
    for i in range(len(model.main)):
        layer = model.main[i]
        if isinstance(layer, nn.Conv2d):
            Hout = int((layerHin+layer.padding[0]*2-layer.dilation[0]*(layer.kernel_size[0]-1)-1)/layer.stride[0]+1)
            Wout = int((layerWin+layer.padding[1]*2-layer.dilation[1]*(layer.kernel_size[1]-1)-1)/layer.stride[1]+1)
            print("Layer "+str(i)+" (Conv2d) output: "+str(Hout)+"x"+str(Wout))
        elif isinstance(layer, nn.ConvTranspose2d):
            Hout = int((layerHin-1)*layer.stride[0]-2*layer.padding[0]+layer.dilation[0]*(layer.kernel_size[0]-1)+layer.output_padding[0]+1)
            Wout = int((layerWin-1)*layer.stride[1]-2*layer.padding[1]+layer.dilation[1]*(layer.kernel_size[1]-1)+layer.output_padding[1]+1)
            print("Layer "+str(i)+" (ConvTranspose2d) output: "+str(Hout)+"x"+str(Wout))
        elif isinstance(layer, nn.Linear):
            Hout = layer.out_features
            Wout = layerWin
        layerHin = Hout
        layerWin = Wout
    print()
    '''
    for i in range(len(model.main)-1, -1, -1):
        layer = model.main[i]
        if isinstance(layer, nn.Conv2d):
            if i==0:
                layerHin = Hin
            else:
                layerHin = desiredHout*2
            kernelSizeShouldBe = calculateKernelSizeForDesiredOutputSize(layer, layerHin, desiredHout)
            print("kernelSize to achieve "+str(desiredHout)+" in layer "+str(i)+" with input "+str(layerHin)+" should be = ", kernelSizeShouldBe)
            desiredHout = layerHin
        elif isinstance(layer, nn.ConvTranspose2d):
            if i==0:
                layerHin = Hin
            else:
                layerHin = (desiredHout+desiredHout%2)/2 
            kernelSizeShouldBe = calculateKernelSizeForDesiredOutputSize(layer, layerHin, desiredHout)
            print("kernelSize to achieve "+str(desiredHout)+" in layer "+str(i)+" with input "+str(layerHin)+" should be = ", kernelSizeShouldBe)
            desiredHout = layerHin
    '''

def computeModel(model, Hin, layersOutputs):
    for layerOutput in layersOutputs:
        layerIndex = layerOutput["layer"]
        layer = model.main[layerIndex]
        desiredHout = layerOutput["output"]
        kernelSizeShouldBe = calculateKernelSizeForDesiredOutputSize(layer, Hin, desiredHout)
        print("kernelSize to achieve "+str(desiredHout)+" in layer "+str(layerIndex)+" with input "+str(Hin)+" should be = ", kernelSizeShouldBe)
        Hin = desiredHout    

def calculateKernelSizeForDesiredOutputSize(layer, Hin, targetH):
    if isinstance(layer, nn.Conv2d):
        #kernelSize = (((targetH - 1)*layer.stride[0] + 1 - Hin - 2*layer.padding[0])/-layer.dilation[0])+1
        kernelSize = (((targetH - 1)*layer.stride[0] + 1 - Hin - layer.padding[0]*2)/-layer.dilation[0])+1
    elif isinstance(layer, nn.ConvTranspose2d):
        kernelSize = ((targetH - (Hin-1)*layer.stride[0] + 2*layer.padding[0] - layer.output_padding[0] - 1)/layer.dilation[0])+1
    return kernelSize
  
def explainDataloader(dataloader):
    print("Description of the input data:")
    X, y = next(iter(dataloader))
    print(X.shape)
    print(y.shape, y.dtype)

def debugHook(name):
    def hook(model, input, output):
        #activation[name] = output.detach()
        print(name, output.shape)
    return hook
    
def registerDebugHook(model):
    for i in range(len(model.main)):
        model.main[i].register_forward_hook(debugHook(model.__class__.__name__+" layer "+str(i)+" output:"))

    

    
    