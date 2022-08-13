# How to backdoor Two-Stream(I3D) Model for Action Recognition on FL ?
 
 ## Overview
Implementation of Two-stream backdoor (**TSB**) and Backdoor combined with adversarial perturbations (**BCAP**).  

#### Download datasets
This code requires two datasets  [UCF-101 ](http://crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) .  

 For convenience, you could download extracted frames and optical flow images from our [aliyunDrive](https://www.aliyundrive.com/drive/).

#### Train backdoored I3D Models on FL
Please follow **Step 1-2**  to train backdoored I3D models and generate attack samples. 

 Alternatively, you can get some pre-trained I3D models on two datasets and  attack samples for triggering backdoor planted in these models from our [aliyunDrive](https://www.aliyundrive.com/drive/) .


#### Configuration files 
There are two important configuration files :

Some optional parameters and models' paths for FL can be set in file `config.json`. 

After  getting UCF-101 and HMDB-51 two datasets, you need to modify their directories  on your system correctly in file `file_path.json`

## Running the code
#### Step 1 (Skippable) -> Train backdoored models on FL
For instance, if you want to train poisoned I3D models by TSB method on UCF-101 dataset, you need to modify 
```json
{
 "data_type" : "ucf-101 ",       "//": "dataset's name 'ucf-101' or 'hmdb-51' ",
 "num_classes" : 101 ,           "//": "ucf-101 contains 101 classes",
 "rgb" : true,                   "//": "train RGB model",
 "flow" : true,                  "//": "train flow Model",
 "poi_rgb" : true,               "//": "Adv would backdoor RGB model",
 "poi_flow" : true`,             "//": "Adv would backdoor flow model",
 "load_pretrained_model": true,  "//": "load model pretrained on Imagenet"
 }
 ```
 in file `config.json`.   

Run `python main.py` to train the I3D models which would be saved in folder `FL_Model`.

#### Step 2  -> Generate attack samples
Now you can utilize  the RGB model trained on UCF-101 , no matter if backdoored or not,   to generate attack samples.

First, please modify the directories where you prefer saving the attack samples in file  `file_path.json`:
```json
{...
 # when adv=true in file `config.json` the RGB attack samples would be saved at "ucf-101_ba_test_frames-adv"
 
"ucf-101_ba_test_frames-poi": "Dir path of poisoned RGB frames（only with trigger) for evaluating RGB model's ASR",  
 
"ucf-101_ba_test_frames-adv": "Dir path of poisoned RGB frames（with trigger and adv-perturbation) for evaluating RGB model's ASR",  
  
"ucf-101_ba_test_flows": "Dir of poisoned flow frames for evaluating Flow model's ASR",
...
}
```

Next, please modify where the model is saved and choose whether to add adversarial perturbations to these samples or not in file `config.json`
```json
{...
"adv": true # add adversarial perturbations to attack samples or not 
"ucf-101-rgb-model":"Dir of RGB model trained on UCF-101",  
"ucf-101-flow-model":"Dir of flow model trained on UCF-101",
...
}

 When 'adv' is true in file `config.json`, the RGB attack samples would be saved at "ucf-101_ba_test_frames-adv". 
 Otherwise, these samples would be saved at "ucf-101_ba_test_frames-poi" .
```
Run (Skippable) 
`cd backdoor_attack`
`python generator.py`
 to generate attack samples.

#### Step 3 -> Evaluate attack effectiveness
(If you have downloaded backdoored models and attack samples from our cloud drive, please follow **Step 2** to modify `file_path.json` and `config.json`)

Run `python Evaluate_ASR.py` to evaluate the attack success rate (ASR) and benign accuracy (BA) on UCF-101.
