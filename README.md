# CVPR21_Transferred_Hash (STILL UNDER CONSTRUCTION!!!)
A PyTorch Implementation of "You See What I Want You to See: Exploring Targeted Black-Box Transferability
Attack for Hash-based Image Retrieval Systems" accepted by CVPR 2021.

## Prepare Enviroments

Needs a 

Basically you will need Python 3 at least and PyTorch 1.0+ to run this program. 

Suggest to install with conda. 

## Dataset
### ImageNet
We use a subset that includes 10% of all classes of the ImageNet following HashNet's implementation, which contains only 100 class. You may download them via [thuml's link](https://drive.google.com/open?id=0B7IzDz-4yH_HSmpjSTlFeUlSS00). 

Remember to modify the path in files in [./HashNet/pytorch/data/imagenet/](./HashNet/pytorch/data/imagenet/) directory. 


## Running
### Preparing Models and Hash Code files.
We provide link for downloading all models and hash code files we have used. Here is the [link](https://drive.google.com/drive/folders/1XkOQNq_mpkGgMuncwk0RdV-PbrvvhVH0?usp=sharing). (**This method is RECOMMENDED.**)

The 'snapshot' folder should be placed under *CVPR21_TransferredHash/AD_attack/HashNet/pytorch/* and the 'save_for_load' folder should be placed under *CVPR21_TransferredHash/AD_attack/HashNet/pytorch/src/*.


You could also do DIY training by:
```
python myTrain.py 

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID       device id to run
  --dataset DATASET     dataset name
  --hash_bit HASH_BIT   number of hash code bits
  --net NET             base network type
  --prefix PREFIX       save path prefix
  --lr LR               learning rate
  --class_num CLASS_NUM
```

Then extract the hash code by:
```
python myExtractCodeLabel.py

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID       device id to run
  --hash_bit HASH_BIT   number of hash code bits
  --snapshot_iter SNAPSHOT_ITER
                        number of iterations the model has
  --dataset DATASET     dataset name
  --net NET             base network type
  --batch_size BATCH_SIZE  batch size to load data
```

 
### Generating Adv.
For NAG(ours method).
```
python myExpForPapers_nag.py --net1 ResNet152 

optional arguments:
  -h, --help            show this help message and exit
  --gpu_id GPU_ID       device id to run
  --dis_method DIS_METHOD
                        distance method
  --adv_method ADV_METHOD
                        adv method
  --var_lambda VAR_LAMBDA
                        lbd to balance loss1 and loss2
  --noise NOISE         noise distribution
  --noise_level NOISE_LEVEL
                        random_noise_level
  --net1 NET1           net1
  --net2 NET2           net2(NOT Necessary and Won't change to results!)
  --l_inf_max L_INF_MAX
                        l_inf_max
  --step_size STEP_SIZE
                        step_size

```
For PGD, DI, DI-Mom (iFGSM, iFGSMDI, miFGSMDI)
```
python myExpGetAdvVulnerable.py --gpu_id 0 --dis_method cW --adv_method iFGSM --net1 ResNet152 
python myExpGetAdvVulnerable.py --gpu_id 0 --dis_method cW --adv_method iFGSMDI --net1 ResNet152 
python myExpGetAdvVulnerable.py --gpu_id 0 --dis_method cW --adv_method miFGSMDI --net1 ResNet152 
    
```


## Contacts and Issues
Yanru Xiao: [yxiao002@odu.edu](mailto:yxiao002@odu.edu)
For any further issues, please oepn a new issue on github. 
