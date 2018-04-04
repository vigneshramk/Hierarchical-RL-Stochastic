#A2C
if [ $1 -eq "0" ]; then
    python main.py --env CartPole-v0 --alg a2c_th --train 1 --render 1

#MAML
elif [ $1 -eq "1" ]; then
    python main.py --env cp-v0 --alg maml_a2c --train 1 --render 1
elif [ $1 -eq "2" ]; then
    python main.py --env Bipedal-v0 --alg maml_a2c --train 1 --render 1
fi
