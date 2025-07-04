import torch
import torch.nn as nn
import timm

class GenderClassifier(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super(GenderClassifier, self).__init__()
        
        # Load the pretrained backbone from timm, By setting num_classes=0, we get a model without the final classification layer.
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        
        # Get the number of input features for the classifier
        # This is specific to the model architecture.
        # For ConvNeXt, it's typically in the 'head' attribute.
        n_features = self.backbone.num_features
        
        # Define a simple linear classifier head
        self.classifier = nn.Linear(n_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        # Pass features through our custom classifier
        output = self.classifier(features)
        
        return output

if __name__ == '__main__':
    import config
    model = GenderClassifier(
        model_name=config.MODEL_NAME, 
        num_classes=config.NUM_CLASSES_GENDER, 
        pretrained=False # Set to False for a quick test without downloading weights
    )
    
    # Create a dummy input tensor
    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE) # (batch_size, channels, H, W)
    output = model(dummy_input)
    
    print(f"Model: {config.MODEL_NAME}")
    print(f"Dummy input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}") # Should be (batch_size, num_classes)
    assert output.shape == (2, config.NUM_CLASSES_GENDER)
    print("Model test passed!")