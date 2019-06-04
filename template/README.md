# sis_competition_task_template

## About this template

You need to submit by using this template for each task.

## Path:

> competition_modules

>> object_detection
    
>>> src (Put your source code here)
    
>>> srv (Design a service used for the task)
    
>>> launch (put your launch file here)
    
>> place_to_box
    
>>> src (Put your source code here)
    
>>> srv (Design a service used for the task)
    
>>> launch (put your launch file here)
    
>> pose_estimate_and_pick
    
>>> src (Put your source code here)
    
>>> srv (Design a service used for the task)
    
>>> launch (put your launch file here)
    
>> robot_navigation
    
>>> src (Put your source code here)
    
>>> srv (Design a service used for the task)
    
>>> launch (put your launch file here)
              
> README.md

> Dockerfile            (You don't need to modify this file)

> run_task.sh           (You don't need to modify this file)

> master_task.launch    (You have to determine which node you need to launch in the task and write in this file)

> docker_build.sh       (If you want to build docker file, please execute/source this shell)


## How to build docker image:

tx2 $ source docker_build.sh

***If docker is already login with other account, please logout first.***

tx2 $ docker logout

***Type your dockerhub's account and password.***

tx2 $ docker login

tx2 $ docker tag sis_competition [dockerhub account]/sis_competition:[task_name]

tx2 $ docker push [dockerhub account]/sis_competition:[task_name]

## How to run

tx2 $ docker run -it [--rm] --name [name] --net host --privileged -v /dev/bus/usb:/dev/bus/usb [dockerhub account]/sis_competition:[task_name]

***If you want to debug by using bash, run this.***

tx2 $ docker run -it [--rm] --name [name] --net host --privileged -v /dev/bus/usb:/dev/bus/usb [dockerhub account]/sis_competition:[task_name] bash

***If you want to debug with visualization, run the following command***

TX2 $ export DISPLAY=:0 && xrandr --fb 1800x900

TX2 $ x11vnc

**After running x11vnc, you'll see available port showed on terminal.**

laptop $ vncviewer  -quality 0 -encodings "tight"  [your tx2 hostname].local:[port]

TX2 $  docker run -it [--rm] --name [name] --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --privileged -v /dev/bus/usb:/dev/bus/usb  [dockerhub account]/sis_competition:[task_name]

***If you want to predict with fcn model, run the following command***

tx2 $  docker run -it [--rm] --name [name] --device=/dev/nvhost-ctrl --device=/dev/nvhost-ctrl-gpu --device=/dev/nvhost-prof-gpu --device=/dev/nvmap --device=/dev/nvhost-gpu --device=/dev/nvhost-as-gpu -v /usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu/tegra --net host -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --privileged -v /dev/bus/usb:/dev/bus/usb -v /home/$USER:/hosthome [dockerhub account]/sis_competition:[task_name] [bash]
