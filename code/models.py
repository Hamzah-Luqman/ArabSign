class VGG16FeaturesExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.fine_tune()
    
    
    def forward(self, x):
        # Shape of x: (batch_size, channels, height, width)
        x = self.vgg16(x)
        return x


    def fine_tune(self):
        for param in self.vgg16.parameters():
            param.requires_grad = False
        
        self.vgg16.classifier = nn.Sequential(*[self.vgg16.classifier[i] for i in range(4)]) # Keeping only till classifier(3) layer. 
