import numpy as np
from PIL import Image

import torchvision

def get_cifar10(root, args, train=True,
                 transform=None,
                 download=False):

    base_dataset = CIFAR10_train(root, args, train=train, transform=transform, download=download)

    noisy_dataset = torchvision.datasets.CIFAR100(root, train=train, download=download)
    if args.ood == 'cifar100':
        print ("OOD: Cifar100")
    elif args.ood == 'svhn':
        print ("OOD: SVHN")
        ood_dataset = torchvision.datasets.SVHN(root, split='train', download=download)
        noisy_dataset.data = np.transpose(ood_dataset.data, (0, 2, 3, 1))
    elif args.ood == 'imagenet':
        print ("OOD: Imagenet")
        noisy_dataset.data = np.load('./data/Imagenet_resize_full.npy')

    noisy_indices = np.random.permutation(len(noisy_dataset.data))
    np.random.shuffle(noisy_indices)
    if args.clean:
        dataset, count = make_clean_dataset(base_dataset, args.percent)
    elif args.ood == 'cifar100' or args.ood == 'svhn' or args.ood == 'imagenet':
        dataset, count = make_noisy_dataset(base_dataset, noisy_dataset, noisy_indices, args.percent)
    else:
        dataset, count = make_noisy_dataset_2(base_dataset, args.percent, args.ood)

    print (f"Clean: {len(dataset) - count} Noisy: {count}")

    if args.close:
        indices = np.random.permutation(len(dataset.data))
        for i, idx in enumerate(indices):
            if i < args.percent * len(dataset.data):
                dataset.targets[idx] = np.random.randint(10)
                dataset.noise_or_not[idx] = 0
        print (f"Clean: {sum(dataset.noise_or_not)} Noisy: {len(dataset) - sum(dataset.noise_or_not)}")
        dataset.make_soft_label()
    return dataset
    

def make_clean_dataset(base_dataset, percent):
    data = []
    label = []
    count = 0
    for i in range(10):
        indices = np.where(np.array(base_dataset.targets) == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j >= percent * len(indices):
                data.append(base_dataset.data[idx])
                label.append(base_dataset.targets[idx])
    base_dataset.data = data
    base_dataset.targets = label
    return base_dataset, count

def make_noisy_dataset(base_dataset, noisy_dataset, noisy_indices, percent):
    count = 0
    data = []
    target = []
    label = []
    noise_or_not = []
    for i in range(10):
        indices = np.where(np.array(base_dataset.targets) == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < percent * len(indices):
                data.append(noisy_dataset.data[noisy_indices[count]])
                count += 1
                label.append(10)
                noise_or_not.append(0)
            else:
                data.append(base_dataset.data[idx])
                label.append(base_dataset.targets[idx])
                noise_or_not.append(1)
            target.append(base_dataset.targets[idx])
            # target.append(label[-1])
    base_dataset.data = data
    base_dataset.targets = target
    base_dataset.gt = label
    base_dataset.noise_or_not = np.array(noise_or_not)
    base_dataset.make_soft_label()
    return base_dataset, count

def make_noisy_dataset_2(base_dataset, percent, ood):
    count = 0
    data = []
    target = []
    label = []
    noise_or_not = []
    for i in range(10):
        indices = np.where(np.array(base_dataset.targets) == i)[0]
        np.random.shuffle(indices)
        for j, idx in enumerate(indices):
            if j < percent * len(indices):
                img = base_dataset.data[idx]
                if ood == 'gaussian':
                    # print ("OOD: Gaussian")
                    noise = np.random.normal(0.2, 1, (32, 32, 3))
                    img = img + 255*noise
                    img = np.clip(img, 0, 255).astype(np.uint8)
                elif ood == 'square':
                    # print ("OOD: Square")
                    img[2:30, 2:30, :] = 0 if np.random.rand(1) < 0.5 else 1
                elif ood == 'reso':
                    # print ("OOD: Resolution")
                    img = Image.fromarray(img)
                    img = img.resize((4,4)).resize((32,32))
                    img = np.array(img)
                data.append(img)
                count += 1
                label.append(10)
                noise_or_not.append(0)
            else:
                data.append(base_dataset.data[idx])
                label.append(base_dataset.targets[idx])
                noise_or_not.append(1)
            target.append(base_dataset.targets[idx])
            # target.append(label[-1])
    base_dataset.data = data
    base_dataset.targets = target
    base_dataset.gt = label
    base_dataset.noise_or_not = np.array(noise_or_not)
    base_dataset.make_soft_label()
    return base_dataset, count

class CIFAR10_train(torchvision.datasets.CIFAR10):

    def __init__(self, root, args=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_train, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.args = args
        self.w = np.ones(len(self.data), dtype=np.float32)
        self.soft_labels = np.zeros((len(self.data), 11), dtype=np.float32)
        self.prediction = np.zeros((len(self.data), 10, 11), dtype=np.float32)

        self.count = 0

    def w_update(self, results):
        self.w = results

    def make_soft_label(self):
        for i, idx in enumerate(range(len(self.data))):
            self.soft_labels[idx][self.targets[idx]] = 0.5
            self.soft_labels[idx][-1] = 0.5

    def label_update(self, results):
        self.count += 1

        # While updating the noisy label y_i by the probability s, we used the average output probability of the network of the past 10 epochs as s.
        idx = (self.count - 1) % 10
        self.prediction[:, idx] = results

        if self.count >= self.args.begin:
            self.soft_labels = self.prediction.mean(axis=1)
            self.targets = np.argmax(self.soft_labels, axis=1).astype(np.int64)

        np.save(f'{self.args.out}/images.npy', self.data)
        np.save(f'{self.args.out}/labels.npy', self.targets)
        np.save(f'{self.args.out}/soft_labels.npy', self.soft_labels)
    
    def reload_label(self):
        self.train_data = np.load(f'{self.args.label}/images.npy')
        self.train_labels = np.load(f'{self.args.label}/labels.npy')
        self.soft_labels = np.load(f'{self.args.label}/soft_labels.npy')
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, soft_target = self.data[index], self.gt[index], self.soft_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, soft_target, index