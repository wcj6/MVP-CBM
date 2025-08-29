# python train_mvpcbm.py -d busi --data-path ./dataset/busi/  --gpu 2



import os
import sys
import time
import math
import pdb
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import balanced_accuracy_score
from optparse import OptionParser
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

import timm
from dataset.dataset import SkinDataset, cmmdDataset, busiDataset, idridDataset, cmDataset, nctDataset, siimDataset
from model import mvpcbm
import utils

DEBUG = False


dataset_dict = {
    'isic2018': SkinDataset,
    'cmmd': cmmdDataset,
    'busi': busiDataset,
    'idrid': idridDataset,
    'cm': cmDataset,
    'nct': nctDataset,
    'siim': siimDataset
}

def train_net(model, config):
    print(config.unique_name)

    train_transforms = copy.deepcopy(config.preprocess)
    train_transforms.transforms.pop(0)
    if model.model_name != 'clip':
        train_transforms.transforms.pop(0)
    train_transforms.transforms.insert(0, transforms.RandomVerticalFlip())
    train_transforms.transforms.insert(0, transforms.RandomHorizontalFlip())
    train_transforms.transforms.insert(0, transforms.RandomResizedCrop(size=(224,224), scale=(0.75, 1.0), ratio=(0.75, 1.33), interpolation=utils.get_interpolation_mode('bicubic')))
    train_transforms.transforms.insert(0, transforms.ToPILImage())

    val_transforms = copy.deepcopy(config.preprocess)
    val_transforms.transforms.insert(0, transforms.ToPILImage())

    trainset = dataset_dict[config.dataset](config.data_path, mode='train', transforms=train_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    trainLoader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=8, drop_last=True)

    valset = dataset_dict[config.dataset](config.data_path, mode='val', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    valLoader = DataLoader(valset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    testset = dataset_dict[config.dataset](config.data_path, mode='test', transforms=val_transforms, flag=config.flag, debug=DEBUG, config=config, return_concept_label=True)
    testLoader = DataLoader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, drop_last=False)


    writer = SummaryWriter(config.log_path+config.unique_name)
    if config.cls_weight == None:
        criterion = nn.CrossEntropyLoss().cuda() 
    else:
        lesion_weight = torch.FloatTensor(config.cls_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=lesion_weight).cuda()
    
    if config.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'adamw':
        optimizer = optim.AdamW([
            {'params': model.get_backbone_params(), 'lr': config.lr * 0.1},
            {'params': model.get_bridge_params(), 'lr': config.lr},
        ])


    BMAC, acc, _, _ = validation(model, valLoader, criterion)
    print('BMAC: %.5f, Acc: %.5f'%(BMAC, acc))

    best_acc = 0

    for epoch in range(config.epochs):
        print('Starting epoch {}/{}'.format(epoch+1, config.epochs))
        batch_time = 0
        epoch_loss_cls = 0
        epoch_loss_concept = 0

        model.train()
        start = time.time()
        
        exp_scheduler = utils.exp_lr_scheduler_with_warmup(optimizer, init_lr=config.lr, epoch=epoch, warmup_epoch=config.warmup_epoch, max_epoch=config.epochs)

        for i, (data, label, concept_label) in enumerate(trainLoader, 0):
            x, target = data.float().cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            optimizer.zero_grad()

            cls_logits, image_logits_dict, sparse_loss = model(x)

            loss_concepts = 0
            idx = 0
            # print(net.concept_token_dict.keys())
            for key in net.concept_token_dict.keys():
                # print(image_logits_dict[key])
                # print(concept_label[:, idx])
                image_concept_loss = F.cross_entropy(image_logits_dict[key], concept_label[:, idx].float())
                loss_concepts += image_concept_loss
                idx += 1
            
            loss_cls = criterion(cls_logits, target)
            loss = loss_cls + loss_concepts / (1000*idx)+ sparse_loss

            loss.backward()
            optimizer.step()

            epoch_loss_cls +=  loss.item()
            epoch_loss_concept=0
            loss_concepts =0
            
            batch_time = time.time() - start
            print(i, 'loss_cls: %.5f, loss_concept: %.5f, batch_time: %.5f' % (loss.item(), loss.item(), batch_time))
        
        print('[epoch %d] epoch loss_cls: %.5f, epoch_loss_concept: %.5f' % (epoch+1, epoch_loss_cls/(i+1), epoch_loss_concept/(i+1) ))

        writer.add_scalar('Train/Loss_cls', epoch_loss_cls/(i+1), epoch+1)
        writer.add_scalar('Train/Loss_concept', epoch_loss_concept/(i+1), epoch+1)


        if not os.path.isdir('%s%s/'%(config.cp_path, config.unique_name)):
            os.makedirs('%s%s/'%(config.cp_path, config.unique_name))
        
        if (epoch+1) % 50 == 0:
            torch.save(model.state_dict(), '%s%s/CP%d.pth'%(config.cp_path, config.unique_name, epoch+1))

        val_BMAC, val_acc, val_loss_cls, val_loss_concept = validation(model, valLoader, criterion)
        writer.add_scalar('Val/BMAC', val_BMAC, epoch+1)
        writer.add_scalar('Val/Acc', val_acc, epoch+1)
        writer.add_scalar('Val/val_loss_cls', val_loss_cls, epoch+1)
        writer.add_scalar('Val/val_loss_concept', val_loss_concept, epoch+1)
        
        test_BMAC, test_acc, test_loss_cls, test_loss_concept = validation(model, testLoader, criterion)
        writer.add_scalar('Test/BMAC', test_BMAC, epoch+1)
        writer.add_scalar('Test/Acc', test_acc, epoch+1)
        writer.add_scalar('Test/test_loss_cls', test_loss_cls, epoch+1)
        writer.add_scalar('Test/test_loss_concept', test_loss_concept, epoch+1) 
                
        lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('LR/lr', lr, epoch+1)


        if val_BMAC >= best_acc:
            best_acc = val_BMAC
            if not os.path.exists(config.cp_path):
                os.makedirs(config.cp_path)
            torch.save(model.state_dict(), '%s%s/best.pth'%(config.cp_path, config.unique_name))
          

        print('save done')
        print('BMAC: %.5f/best BMAC: %.5f, Acc: %.5f'%(val_BMAC, best_acc, val_acc))


        
def validation(model, dataloader, criterion):
    
    net = model
    net.eval()

    losses_cls = 0
    losses_concepts = 0

    pred_list = np.zeros((0), dtype=np.uint8)
    gt_list = np.zeros((0), dtype=np.uint8)

    with torch.no_grad():
        for i, (data, label, concept_label) in enumerate(dataloader):
            data, label = data.cuda(), label.long().cuda()
            concept_label = concept_label.long().cuda()
            cls_logits, image_logits_dict,kl_loss = net(data)
            loss_cls = criterion(cls_logits, label)+kl_loss
            losses_cls += loss_cls.item()
            _, label_pred = torch.max(cls_logits, dim=1)
            pred_list = np.concatenate((pred_list, label_pred.cpu().numpy().astype(np.uint8)), axis=0)
            gt_list = np.concatenate((gt_list, label.cpu().numpy().astype(np.uint8)), axis=0)
    
    BMAC = balanced_accuracy_score(gt_list, pred_list)
    correct = np.sum(gt_list == pred_list)
    acc = 100 * correct / len(pred_list)

    return 100*BMAC, acc, losses_cls/(i+1), losses_concepts/(i+1)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
            help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=128,
            type='int', help='batch size')
    parser.add_option('--warmup_epoch', dest='warmup_epoch', default=5, type='int')
    parser.add_option('--optimizer', dest='optimizer', default='adamw', type='str')
    parser.add_option('-l', '--lr', dest='lr', default=0.0001, 
            type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False,
            help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path',
            default='./checkpoint/', help='checkpoint path')
    parser.add_option('-o', '--log-path', type='str', dest='log_path', 
            default='./log/', help='log path')
    parser.add_option('-m', '--model', type='str', dest='model',
            default='mvpcbm', help='use which model')
    parser.add_option('--linear-probe', dest='linear_probe', action='store_true', help='if use linear probe finetuning')
    parser.add_option('-d', '--dataset', type='str', dest='dataset', 
            default='isic2018', help='name of dataset')
    parser.add_option('--data-path', type='str', dest='data_path', 
            default='./dataset/isic2018/', help='the path of the dataset')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name',
            default='test', help='name prefix')
     
    parser.add_option('--flag', type='int', dest='flag', default=2)
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
   

    (config, args) = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    config.log_path = config.log_path + config.dataset + '/'
    config.cp_path = config.cp_path + config.dataset + '/'
    
    print('use model:', config.model)
    
    num_class_dict = {
        'isic2018': 7,
        'cmmd': 2,
        'idrid':5,
        'busi':3,
        'cm':2,
        'edema':2,
        'nct':9,
        'siim':2
    }
   
    cls_weight_dict = {
        'isic2018': [1.2855, 0.2134, 2.7835, 4.3753, 1.3018,12.4410, 10.0755], 
        'busi': [3.1579,  0.9611, 2.0635], 
        'cmmd': [3.7037, 1.3776],
        # miniddsm
        # 'cmmd': [1.006, 0.994],
        'idrid': [0.614,4.128, 0.614, 1.11, 1.66 ],
        'cm': [1.985, 0.668],
        'edema': [1.206, 0.854],
        'nct':[0.6, 0.94, 2.35, 1.26, 0.77, 1.35, 1.08, 1.89, 0.65],
        'siim': [0.6423 , 2.2568]
    }
    concept_dict = {
    'isic2018': {
    'color': ['highly variable, often with multiple colors (black, brown, red, white, blue)',   'uniformly tan, brown, or black',  'translucent, pearly white, sometimes with blue, brown, or black areas',   'red, pink, or brown, often with a scale', 'light brown to black',   'pink brown or red', 'red, purple, or blue'],
    'shape': ['irregular', 'round', 'round to irregular', 'variable'],
    'border': ['often blurry and irregular', 'sharp and well-defined', 'rolled edges, often indistinct'],
    'dermoscopic patterns': ['atypical pigment network, irregular streaks, blue-whitish veil, irregular',  'regular pigment network, symmetric dots and globules',  'arborizing vessels, leaf-like areas, blue-gray avoid nests',  'strawberry pattern, glomerular vessels, scale',   'cerebriform pattern, milia-like cysts, comedo-like openings',    'central white patch, peripheral pigment network', 'depends on type (e.g., cherry angiomas have red lacunae; spider angiomas have a central red dot with radiating legs'],
    'texture': ['a raised or ulcerated surface', 'smooth', 'smooth, possibly with telangiectasias', 'rough, scaly', 'warty or greasy surface', 'firm, may dimple when pinched'],
    'symmetry': ['asymmetrical', 'symmetrical', 'can be symmetrical or asymmetrical depending on type'],
    'elevation': ['flat to raised', 'raised with possible central ulceration', 'slightly raised', 'slightly raised maybe thick']
},
    "cmmd": { "Mass Shape": [
        "Round/Oval: Smooth, well-defined edges",
        "Irregular: Asymmetrical with no definable shape",
        "Spiculated: Star-shaped with radiating lines"
    ], "Mass Margin": [
        "Circumscribed: Clear, well-defined borders",
        "Ill-defined: Blurred, indistinct borders",
        "Spiculated: Spiky, radiating margins"
    ], "Mass Density": [
        "Low Density (Radiolucent)",
        "Isodense: Similar to surrounding tissue",
        "High Density (Radiopaque)"
    ],  "Calcifications": [
        "Absent: No calcifications present",
        "Benign Calcifications: Macrocalcifications with smooth shapes",
        "Suspicious Calcifications: Clustered microcalcifications with irregular patterns"
    ],   "Architectural Distortion": [
        "None: Normal breast architecture",
        "Minimal: Slight distortion without associated mass",
        "Significant: Noticeable distortion often linked to an underlying mass"
    ],  "Asymmetry": [
        "None: Symmetrical breast tissue",
        "Mild: Slight differences in breast tissue density or shape",
        "Marked: Pronounced differences with associated suspicious features"
    ]
},

    "idrid": {
        "Color": [
            "Small red dots",
             "Increased redness",
             "Deeper red hue",
            "Dark red to purple",
             "Very dark red to obscured"
        ],
        "Shape": [
             "Circular or slightly irregular",
             "Irregular shapes emerging",
            "More irregular and varied shapes",
            "Highly irregular and varied shapes",
             "Highly distorted, irregular forms"
        ],
        "Border": [
             "Well-defined",
             "Slightly blurred",
             "Partially obscured",
             "Poorly defined",
             "Obscured by neovascularization or fibrous tissue"
        ],
        "Texture": [
            "Smooth",
             "Slightly granular",
            "Varied textures",
             "Rough and heterogeneous",
             "Irregular, coarse textures"
        ],
        "Symmetry": [
             "Typically symmetrical",
             "Mild asymmetry possible",
             "Asymmetrical distribution",
            "Highly asymmetrical and extensive",
             "Highly asymmetrical"
        ],
        "Elevation": [
             "Flat",
            "Mostly flat",
             "Mostly flat",
             "Some elevation may occur",
            "Possible elevation or distortion"
        ]
    },
    "busi": {
    "Echogenicity": [
        "Anechoic (completely dark, fluid-filled)",
        "Hypoechoic (slightly darker than surrounding tissue)",
        "Isoechoic (similar echogenicity to surrounding tissue)",
        "Hyperechoic (brighter than surrounding tissue)",
        "Markedly hyperechoic with shadowing"
    ],
    "Shape": [
        "Oval (smooth, regular edges)",
        "Round (circular, symmetric)",
        "Lobulated (irregular but with smooth transitions)",
        "Angular (sharp, distinct edges)",
        "Irregular (no definable shape, spiculated)"
    ],
    "Margin": [
        "Circumscribed (clear, well-defined borders)",
        "Fuzzy (slightly blurred borders)",
        "Microlobulated (small, multiple lobules at the edges)",
        "Obscured (poorly defined borders)",
        "Spiculated (spiky, radiating lines from the margin)"
    ],
    "Orientation": [
        "Parallel (aligned with the skin surface)",
        "Not parallel (perpendicular or non-aligned)",
        "Antiparallel (tilted away from the skin surface)",
        "Complex orientation with mixed alignment",
        "Variable orientation with no consistent pattern"
    ],
    "Posterior_Features": [
        "Enhancement (increased brightness behind the lesion)",
        "No significant change",
        "Shadowing (reduced echogenicity behind the lesion)",
        "Reverberation artifacts",
        "Echogenic foci with shadowing"
    ],
    "Surrounding_Tissue": [
        "Normal echotexture with no surrounding abnormalities",
        "Minimal surrounding tissue changes",
        "Increased echogenicity in surrounding tissue",
        "Decreased echogenicity or fibrosis around the lesion",
        "Significant architectural distortion of surrounding tissue"
    ]
},
   "cm": {
    "Heart Silhouette": [
        "Oval shape with smooth borders",
        "Slightly globular shape with minor irregularities",
        "Moderately globular or elongated shape",
        "Significantly globular or irregular shape with pronounced borders",
        "Highly irregular or distorted shape with unclear borders"
    ],
    "Mediastinal Shift": [
        "No shift; mediastinum is central",
        "Minor shift towards one side without significant displacement",
        "Moderate shift towards one side with noticeable displacement",
        "Significant shift towards one side affecting mediastinal structures",
        "Severe shift causing substantial displacement of mediastinal structures"
    ],
    "Pulmonary Congestion": [
        "No signs of pulmonary congestion",
        "Mild vascular congestion without interstitial markings",
        "Moderate vascular congestion with some interstitial markings",
        "Severe vascular and interstitial congestion with prominent markings",
        "Extensive pulmonary congestion with alveolar filling patterns"
    ],
    "Associated Findings": [
        "No associated findings",
        "Mild pleural effusion or slight pulmonary edema",
        "Moderate pleural effusion, cardiogenic pulmonary edema",
        "Significant pleural effusion, extensive pulmonary edema",
        "Massive pleural effusion, acute respiratory distress signs"
    ]
},
   "nct": {
    "Color": [
        "Light pink to yellow",
        "Variable, depending on surrounding tissues",
        "Dark brown to black",
        "Dark blue to purple",
        "Light blue to clear",
        "Deep pink to reddish-brown",
        "Light pink to reddish",
        "Darker pink to brown"
    ],
    "Texture": [
        "Soft, homogeneous",
        "Homogeneous or heterogeneous without specific features",
        "Irregular, clumped",
        "Compact, dense",
        "Gelatinous, amorphous",
        "Striated, elongated cells",
        "Layered, glandular",
        "Fibrotic, dense",
        "Pleomorphic, hyperchromatic nuclei"
    ],
    "Shape": [
        "Rounded or lobulated clusters",
         "Fragmented and amorphous",
         "Small, round nuclei with scant cytoplasm",
         "Variable, often forming pools or secretory droplets",
        "Spindle-shaped nuclei aligned in parallel bundles",
         "Regular crypt structures with uniform size",
         "Irregular, interwoven fibers",
         "Irregular glandular structures, loss of uniform crypt shape"
    ],
    "Size": [
         "Variable, often dispersed throughout the tissue",
        "Small to medium-sized clusters",
        "Consistently small across samples",
       "Variable, can be widespread or localized",
         "Uniform, forming fascicles",
         "Consistent glandular units throughout",
         "Variable, often expanding around tumor cells",
         "Variable, with increased nuclear size and mitotic figures"
    ],
    "Additional Features": [
        "Contains lipid droplets and clear cytoplasm",
         "Non-specific areas without identifiable structures",
         "Presence of necrotic material and cellular fragments",
         "High nucleus-to-cytoplasm ratio, lack of prominent nucleoli",
       "Presents as extracellular material, often surrounding glands",
         "Presence of muscle fibers with characteristic striations",
         "Presence of goblet cells and well-organized epithelial layers",
         "Increased collagen deposition, myofibroblasts, and altered extracellular matrix",
         "Disorganized architecture, glandular crowding, mucin production, and invasive growth patterns"
    ]
},
"siim": {
    "Radiolucency": [
        "Distinct radiolucent area indicating air presence",
        "Large radiolucent space in the apex of the lung",
        "Radiolucent zone typically located at the periphery of the lung fields"
    ],
    "Pleural Line": [
        "Sharp and clear pleural line separating lung tissue from air",
        "Pleural line may appear as a thin, straight or irregular line",
        "Absence of lung markings beyond the pleural line"
    ],
    "Lung Markings": [
        "No visible lung markings beyond the pleural line",
        "Sudden termination of lung markings at the pleural line",
        "Absence of normal lung parenchymal structures beyond the pleural line"
    ],
    "Mediastinal Shift": [
        "Shift of mediastinal structures towards the opposite side in tension pneumothorax",
        "Displacement of the heart shadow and trachea towards the unaffected side",
        "Mediastinal shift observed in severe cases as an indication of tension pneumothorax"
    ],
    "Lung Edge": [
        "Visible lung edge with a clear boundary against the chest wall",
        "Lung edge may appear serrated or smooth",
        "No extension of normal lung tissue beyond the edge"
    ],
    "Size": [
        "Small pneumothorax: < 3 cm between the pleural line and chest wall at the level of the hilum",
        "Moderate pneumothorax: 3-6 cm distance",
        "Large pneumothorax: > 6 cm distance",
        "Pneumothorax size can be assessed by measuring the distance between the pleural line and chest wall"
    ],
    "Additional Features": [
        "Compression or collapse of lung tissue in the affected area",
        "Blurring of the cardiac silhouette if significant",
        "Presence of subcutaneous emphysema",
        "Visible diaphragmatic depression on the affected side",
        "In lateral views, elevated hemidiaphragm on the affected side"
    ]
}
}
    
    config.cls_weight = cls_weight_dict[config.dataset]
    config.num_class = num_class_dict[config.dataset]
    concept_list = concept_dict[config.dataset]

    net = mvpcbm(concept_list=concept_list, model_name='biomedclip', config=config)

    # We find using orig_in21k vit weights works better than biomedclip vit weights
    # Delete the following if want to use biomedclip weights
    vit = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=config.num_class)
    vit.head = nn.Identity()
    net.model.visual.trunk.load_state_dict(vit.state_dict())

    if config.load:
        net.load_state_dict(torch.load(config.load))
        print('Model loaded from {}'.format(config.load))

    net.cuda()
    # summary = net.summary()
    # print(summary)

    train_net(net, config)

    print('done')

