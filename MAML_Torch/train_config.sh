#lnQ without replay memory
if [ $1 -eq "0" ]; then
    python main.py --env CartPole-v0 --alg lnQ_NR --train 1 --render 1
elif [ $1 -eq "1" ]; then
    python main.py --env MountainCar-v0 --alg lnQ_NR --train 1 --render 1

#lnQ with replay memory
elif [ $1 -eq "2" ]; then
    python main.py --env CartPole-v0 --alg lnQ --train 1 --render 1
elif [ $1 -eq "3" ]; then
    python main.py --env MountainCar-v0 --alg lnQ --train 1 --render 1

#DQN
elif [ $1 -eq "4" ]; then
    python main.py --env CartPole-v0 --alg DQN --train 1 --render 1
elif [ $1 -eq "5" ]; then
    python main.py --env MountainCar-v0 --alg DQN --train 1 --render 1
elif [ $1 -eq "6" ]; then
    python main.py --env SpaceInvaders-v0 --alg DQN --train 1 --render 1

#DuQN
elif [ $1 -eq "7" ]; then
    python main.py --env CartPole-v0 --alg DuQN_tf --train 1 --render 1
elif [ $1 -eq "8" ]; then
    python main.py --env MountainCar-v0 --alg DuQN_tf --train 1 --render 1

elif [ $1 -eq "9" ]; then
    python main.py --env CartPole-v0 --alg DuQN_th --train 1 --render 1

#A2C
elif [ $1 -eq "10" ]; then
    python main.py --env CartPole-v0 --alg a2c_th --train 1 --render 1

#MAML
elif [ $1 -eq "11" ]; then
    python main.py --env cp-v0 --alg maml_a2c --train 1 --render 1
elif [ $1 -eq "12" ]; then
    python main.py --env Bipedal-v0 --alg maml_a2c --train 1 --render 1
fi
