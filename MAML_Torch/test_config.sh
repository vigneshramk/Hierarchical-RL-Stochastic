if [ $1 -eq "0" ]; then
    python main.py --env CartPole-v0 --alg a2c_th --train 0 --render 1 --actor_model $2 --critic_model $3

#MAML
elif [ $1 -eq "1" ]; then
    python main.py --env cp-v0 --alg maml_a2c --train 0 --render 1 --actor_model $2 --critic_model $3
elif [ $1 -eq "2" ]; then
    python main.py --env Bipedal-v0 --alg maml_a2c --train 0 --render 1 --actor_model $2 --critic_model $3
fi
