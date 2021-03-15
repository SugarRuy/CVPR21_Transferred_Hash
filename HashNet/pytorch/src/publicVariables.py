# -*- coding: utf-8 -*-

# Default Iter for doing the experiments
iters_list = {"ResNet18":33000, "ResNet34":20000, "ResNet50": 10000, "ResNet101": 47000,
              "ResNet152":94000,  "ResNext101_32x4d":94000, 'Inc_v3': 97000, 'VGG16BN':102000,
              "VGG19BN":100000, "DenseNet161": 100000, "SEResNet50":150000, "SEResNet101":190000,
              "SENet154": 60000} # disable "Inc_v4":36000,



data_list_path = {'imagenet': {'database': "../data/imagenet/database.txt", \
                               'test': "../data/imagenet/test.txt", \
                               'train_set1': "../data/imagenet/train.txt", \
                               'train_set2': "../data/imagenet/train.txt", \
                               'train': "../data/imagenet/train.txt"}, \
                  'places365': {'database': "../data/places365_standard/database.txt", \
                                'test': "../data/places365_standard/val.txt", \
                                'train_set1': "../data/places365_standard/train.txt", \
                                'train_set2': "../data/places365_standard/train.txt", \
                                'train': "../data/places365_standard/train.txt"} \
                  }

layer_index_value_list = {"ResNext101_32x4d":[464, 482, 487, 489, 497, 502, 505, 512],\
                          "ResNet152":[361, 369, 377, 393, 405, 413, 425]}

# The network trained and saved as 'state_dict.pth' pattern
new_networks = ['ResNext101_32x4d', 'SEResNext50_32x4d', 'SEResNet50', 'SENet154', 'SEResNet101',
                'ResNet34', 'VGG16BN']