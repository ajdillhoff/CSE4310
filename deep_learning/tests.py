import torch

import FasterRCNN

# Validate output of RegionProposalNetwork
def test_rpn():
    rpn = FasterRCNN.RegionProposalNetwork(2048, 512, scales=[32, 64, 128], ratios=[0.5, 1, 2], feature_strides=[16])
    x = torch.rand(1, 2048, 32, 32)
    rpn.eval()
    scores, bbox_pred, anchors = rpn(x, [(32, 32)])

    print(scores)


if __name__ == '__main__':
    test_rpn()