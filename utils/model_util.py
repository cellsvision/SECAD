import torch

def get_model(model_name, pretrained=True, num_classes=4, need_head=True):
    # torch.hub.set_dir(r'/tmp/.cache/torch/hub')
    if model_name == 'cbam18':
        from models.CBAM import resnet18_cbam

        model = resnet18_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'cbam34':
        from models.CBAM import resnet34_cbam
        model = resnet34_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'cbam50':
        from models.CBAM import resnet50_cbam
        model = resnet50_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'cbam2_18':
        from models.CBAM2 import resnet18_cbam
        model = resnet18_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'cbam2_34':
        from models.CBAM2 import resnet34_cbam
        model = resnet34_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'cbam2_50':
        from models.CBAM2 import resnet50_cbam
        model = resnet50_cbam(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'resnet152':
        from models.resnet import resnet152
        model = resnet152(pretrained=pretrained, num_classes=num_classes, need_head=need_head)
    elif model_name == 'efficientnet_b3':
        # model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b3', pretrained=True)
        from models.efficientnet import EfficientNet
        if pretrained:
            model = EfficientNet.from_pretrained('efficientnet-b3',num_classes=num_classes)
        else:
            model = EfficientNet.from_name('efficientnet-b3',num_classes=num_classes)
    elif model_name == 'lstm':
        from models.bilstm_attention import BiLSTM_Attention

        model = BiLSTM_Attention(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'gcn_s':
        from models.gcn import pvig_s_224_gelu
        model = pvig_s_224_gelu(pretrained=pretrained, num_classes=num_classes)
    elif model_name == 'convnext_small':
        from models.convnext import convnext_small
        model = convnext_small(pretrained=True,in_22k = True,num_classes=num_classes)
    elif model_name == 'convnext':
        from models.convnext import convnext_base
        model = convnext_base(pretrained=True,in_22k = True,num_classes=num_classes)
    elif model_name == 'convnext_large':
        from models.convnext import convnext_large
        model = convnext_large(pretrained=True,in_22k = True,num_classes=num_classes)
    elif model_name == 'convnext_xlarge':
        from models.convnext import convnext_xlarge
        model = convnext_xlarge(pretrained=True,in_22k = True,num_classes=num_classes)
    elif model_name == 'transfg':
        from models.transfg import CONFIGS,VisionTransformer
        import numpy as np
        cfg = CONFIGS['ViT-B_32']
        model = VisionTransformer(cfg, img_size=448)
        model.load_from(np.load('models/imagenet21k+imagenet2012_ViT-B_32.npz'))
    elif model_name == 'moganet_base':
        from models.moganet import moganet_base
        model = moganet_base(pretrained=False,num_classes=num_classes)
    elif model_name == 'convnextv2_base':
        from models.convnextv2 import convnextv2_base
        model = convnextv2_base(num_classes=num_classes)
    elif model_name == 'racnn':
        from models.RACNN import RACNN
        model = RACNN(num_classes=num_classes)
    elif model_name == 'racnn2':
        from models.RACNN2 import RACNN
        model = RACNN(num_classes=num_classes)
    elif model_name == 'racnn3':
        from models.RACNN3 import RACNN
        model = RACNN(num_classes=num_classes)
    elif model_name == 'racnn4':
        from models.RACNN4 import RACNN
        model = RACNN(num_classes=num_classes)
    elif model_name == 'racnn5':
        from models.RACNN5 import RACNN
        model = RACNN(num_classes=num_classes)
    else:
        raise Exception('Model %s not exists.' % model_name)
    return model

if __name__ == '__main__':
    model = get_model('efficientnet_b3')
    print()
