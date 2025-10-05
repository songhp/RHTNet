import torch
import platform
import torchvision
from torch.utils.data import DataLoader, DistributedSampler


def data_loader_new(args, root='./', is_test=False):
    kwopt = {'num_workers': 16, 'pin_memory': True}
    w_size, h_size = int(16 * 8), int(16 * 8)
    trn_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    test_set5_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])
    test_set14_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])

    # Transformers for BSDS
    test_bsds_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])
    test_compare_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((w_size, h_size)),
        torchvision.transforms.ToTensor(),
    ])

    trn_dataset = torchvision.datasets.ImageFolder(root + 'dataset/test', transform=trn_transforms)
    test_bsds = torchvision.datasets.ImageFolder(root + 'dataset/test', transform=test_bsds_transforms)
    test_set5 = torchvision.datasets.ImageFolder(root + 'dataset/set5', transform=test_set5_transforms)
    test_set14 = torchvision.datasets.ImageFolder(root + 'dataset/set14', transform=test_set14_transforms)
    compare = torchvision.datasets.ImageFolder(root + 'dataset/set5', transform=test_compare_transforms)

    test_loader_bsds = DataLoader(test_bsds, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_set5 = DataLoader(test_set5, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_set14 = DataLoader(test_set14, batch_size=1, shuffle=False, **kwopt, drop_last=False)
    test_loader_compare = DataLoader(compare, batch_size=1, shuffle=False, **kwopt, drop_last=False)

    if is_test:
        return None, test_loader_bsds, test_loader_set5, test_loader_set14, test_loader_compare

    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, **kwopt,
                            drop_last=False)

    return trn_loader, test_loader_bsds, test_loader_set5, test_loader_set14
