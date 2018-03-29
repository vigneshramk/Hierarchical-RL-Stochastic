
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str, help='gym environment name eg. CartPole-v0, MountainCar-v0, SpaceInvaders-v0')
    parser.add_argument('--alg',dest='alg', type=str,default='DQN', help='Choose the algorithm to use between lnQ_NR, lnQ, DQN, DuQN_tf and DuQN_th')
    parser.add_argument('--render',dest='render',type=int,default=0, help='Choose whether to render or not')
    parser.add_argument('--train',dest='train',type=int,default=1, help='Choose whether to train or test')
    parser.add_argument('--model',dest='model_file',type=str, help='Specify the model_file to load and test')
    parser.add_argument('--actor_model', dest='actor_model_file', type=str,
                        help='Specify the actor model_file to load and test')
    parser.add_argument('--critic_model', dest='critic_model_file', type=str,
                        help='Specify the critic model_file to load and test')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    if args.alg == 'lnQ_NR':
        import lnQ_NR_keras as lnQ_NR
        print('\n\n\nRunning Linear Q Network without replay...\n\n\n')
        lnQ_NR.lnQ_NR_main(args)
    elif args.alg == 'lnQ':
        import lnQ_keras as lnQ
        print('\n\n\nRunning Linear Q Network with replay...\n\n\n')
        lnQ.lnQ_main(args)
    elif args.alg == 'DQN':
        import DQN_keras as DQN
        print('\n\n\nRunning DQN with replay...\n\n\n')
        DQN.DQN_main(args)
    elif args.alg == 'DuQN_tf':
        import DuelingQ_keras as DuQN_tf
        print('\n\n\nRunning Dueling Q Network replay using tensorflow...\n\n\n')
        DuQN_tf.DuQN_main(args)
    elif args.alg == 'DuQN_th':
        import DuelingQ_torch as DuQN_th
        print('\n\n\nRunning Dueling Q Network replay using torch...\n\n\n')
        DuQN_th.DuQN_main(args)
    elif args.alg == 'SI5':
        import SI5 as SI5
        print('\n\n\nRunning SI5 Q Network replay...\n\n\n')
        SI5.SI5_main(args)
    elif args.alg == 'a2c_th':
        import a2c_torch
        print('\n\n\nRunning A2C using torch...\n\n\n')
        a2c_torch.a2c_main(args)
    elif args.alg == 'maml_a2c':
        import maml_a2c
        print('\n\n\nRunning MAML...\n\n\n')
        maml_a2c.maml_main(args)



