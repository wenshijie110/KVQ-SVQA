import torch
import torch.nn as nn
from convnext import convnext_tiny
from swin_transformer_V2 import SwinTransformerV2


class simpleVQAHead(nn.Module):

    def __init__(self, in_channels=4096+2048+1024+2048+256):
        super().__init__()

        self.quality = nn.Sequential(
            nn.Linear(in_channels, in_channels//4),
            nn.Linear(in_channels // 4, in_channels // 32),
            nn.Linear(in_channels // 32, 1),
        )

    def forward(self, x):
        x = self.quality(x)
        x = torch.mean(x, dim=1)  # frame avg
        return x


class VQA_Network(nn.Module):

    def __init__(self, ):
        super().__init__()
        self.backbone = SwinTransformerV2(img_size=256, window_size=8)
        self.head = simpleVQAHead(in_channels=4224 + 2304 + 2688)
        self.sal_backbone = convnext_tiny(pretrained=True, in_22k=True)

        state_dict = torch.load('swinv2_tiny_patch4_window8_256.pth', map_location="cpu")
        self.backbone.load_state_dict(state_dict['model'], strict=False)

    def forward(self, input_feature):

        x_3d_features = input_feature['feat'].to(input_feature['frame_feature'].device)
        x_3d_features_size = x_3d_features.shape
        x_3d_features = x_3d_features.view(-1, x_3d_features_size[2])

        feature = input_feature['sal_feature']
        b, c, t, h, w = feature.size()   # 8,3,16,256,256
        feature = feature.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()   # 128,3,256,256
        out_feature = self.backbone(feature)

        sal_feature = input_feature['frame_feature']  # 8,3,16,224,224
        sal_feature = sal_feature.permute(0, 2, 1, 3, 4).flatten(0, 1).contiguous()
        sal_feature = self.sal_backbone(sal_feature)

        x = torch.cat((out_feature, x_3d_features, sal_feature), dim=1)
        x = x.view(b, t, -1)

        scores = self.head(x)

        return scores





