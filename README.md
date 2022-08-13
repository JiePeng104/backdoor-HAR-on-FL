# How to backdoor Two-Stream(I3D) Model for Action Recognition on FL ?
 
 ## Overview
Implementation of Two-stream backdoor (**TSB**) and Backdoor combined with adversarial perturbations (**BCAP**).  

### Download datasets
This code requires two datasets  [UCF-101 ](http://crcv.ucf.edu/data/UCF101.php) and [HMDB-51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) .  

 For convenience, you could download extracted frames and optical flow images from our [aliyunDrive](https://www.aliyundrive.com/drive/).

### Train backdoored I3D Models on FL
If you want to train backdoored I3D models and generate attack samples, please follow **Step 1-2** .  Alternatively, you can get some pre-trained I3D models on two datasets and  attack samples for triggering backdoor planted in these models from our [aliyunDrive](https://www.aliyundrive.com/drive/) .


### Configuration files 
There are two important configuration files.  Some optional parameters and models' paths for FL can be set in file `config.json`. After you get UCF-101 and HMDB-51 two datasets, you need to set their directories on your system in file `file_path.json`

## Running the code
### Step 1 -> Get RGB frames and optical flow images
