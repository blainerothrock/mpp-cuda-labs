# Massively Parallel Programming with CUDA 

## Authors
Blaine Rothrock, Nayan Mehta, Grant Gasser

## Connecting to lab machines and running
* `ssh` into lab machine with NVIDA GPU
* Run `source /usr/local/cuda-5.0/cuda-env.csh`
* cd into the `lab1/` folder, then run `make`
* go back a few directories and cd into `bin/linux/release` and run the exectuable: `./lab1` 

## Setup CLion
* Tools->Start SSH Session...-> enter credentials
* Run `cmd` + `,`, then Build, Execution, Deployment->Deployment, choose **Type** as SFTP, enter credentials, **Autodetect** root path. 
* Type in "remote host" in the help/search bar and choose **Browse Remote Host**, this will show the file structure of the remote machine

## Lab Assignments
1. Matrix Multiplication
2. Tiled Matrix Multiplication
3. Histograms
4. Parallel Prefix Scan for Large Arrays
