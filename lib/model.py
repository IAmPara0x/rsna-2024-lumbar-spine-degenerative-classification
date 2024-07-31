
import timm
import torch
import torch.nn as nn


class AvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self,x):
        r = list(x.shape)[:-2]
        r.append(-1)
        x = x.reshape(tuple(r)).mean(dim=-1)
        return x

class DetectionModel(nn.Module):
    def __init__(self, model_name, n_classes, coord_dim, hidden_dim, pretrained=True, features_only=True):
        super().__init__()
        
        # TODO: Find a way to not hardcode these variables
        self.img_features  = 512
        self.img_features_dim = 2048
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.coord_dim = coord_dim
    
        self.initial_conv = nn.Conv2d(1,3, (3,3), stride=(1,1), padding=(1,1))
        self.feature_extractor =  timm.create_model( model_name , pretrained=pretrained , features_only=features_only, out_indices=[-1] )
    
        self.predictors = nn.Sequential(
            nn.Conv2d(self.img_features, self.img_features, (3,3), stride=(1,1), padding=(1,1)),
            nn.SiLU(),
            nn.Conv2d(self.img_features, self.img_features_dim, (1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(self.img_features_dim),
            AvgPool(),
            nn.Linear(self.img_features_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_classes)
        )
    
        self.location_predictor = nn.Sequential(
            nn.Conv2d(self.img_features, self.img_features, (3,3), stride=(1,1), padding=(1,1)),
            nn.SiLU(),
            nn.Conv2d(self.img_features, self.img_features, (3,3), stride=(1,1), padding=(1,1)),
            nn.SiLU(),
            nn.Conv2d(self.img_features, self.img_features_dim, (1,1), stride=(1,1), bias=False),
            nn.BatchNorm2d(self.img_features_dim),
            AvgPool(),
            nn.Linear(self.img_features_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.coord_dim * self.n_classes)
        )


    def forward(self, x, freeze_conv=False):
        x = self.initial_conv(x)

        if freeze_conv:
            with torch.no_grad():
                x = self.feature_extractor(x)[0]
        else:
            x = self.feature_extractor(x)[0]
            
        class_preds = self.predictors(x)
        location_preds = self.location_predictor(x)
        return class_preds, location_preds


class CenterNet():
    pass
