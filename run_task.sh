#!/bin/bash

set -e

source ~/sis_mini_competition_2018/catkin_ws/devel/setup.sh

#roslaunch sis_arm_planning master_task.launch
roslaunch sis_arm_planning manipulation_tx2.launch
