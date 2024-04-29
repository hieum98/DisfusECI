import argparse
import time


def create_argument_parser():
    """Defines a parameter parser for all of the arguments of the application.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ###### Data parameters ##########
    parser.add_argument("--data_name", type=str, default="ESL",
                        help="Specifies which dataset to use, see README for options")
    parser.add_argument("--inter", action='store_true', default=False,
                        help="Specifies which type of data use. Note: only apply for ESL corpus")
    parser.add_argument("--intra", action='store_true', default=False,
                        help="Specifies which type of data use. Note: only apply for ESL corpus")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size in training, validating, and testing")
    parser.add_argument("--label_max_len", type=int, default=16,
                        help="Max length of label sentences")
    
    ###### Model parameters ##########
    parser.add_argument("--model_name", type=str, default="roberta-large",
                        help="Specifies which pretrained model to use, see README for options")
    parser.add_argument("--block_type", type=str, default="in-context",
                        choices=["in-context", "adaLN-Zero"],
                        help="Specifies which block type of model to use, see README for options")
    parser.add_argument("--diff_depth", type=int, default=4,
                        help="Number layer of diffusion model")
    parser.add_argument('--use_diffusion', action='store_true', default=False, 
                        help='Use diffusion latents as augmented features')

    ###### Training and inferencing parameters #######
    parser.add_argument("--diff_lr", type=float, default=2e-5,
                        help="Learning rate using to train diffusion model")
    parser.add_argument("--encoder_lr", type=float, default=2e-5,
                        help="Learning rate using to train encoder model")
    parser.add_argument("--head_lr", type=float, default=2e-5,
                        help="Learning rate using to train task's header layers")
    parser.add_argument("--diffusion_train_step", type=int, default=1000,
                        help="Number of diffusion steps used to train the model")
    parser.add_argument("--diffusion_inference_step", type=int, default=100,
                        help="The number of diffusion steps used when generating samples with a pre-trained model.")
    parser.add_argument("--num_epochs", type=int, default=20,
                        help="Number of epochs used to train the model")
    parser.add_argument("--encoder_warm_up", type=int, default=10,
                        help="Number epoch used to warm up the encoder model")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of gradient accumulation steps used to train the model")
    
    ###### MISC parameters ######
    parser.add_argument("--config_file", type=str, default='/sensei-fs/users/daclai/hieu/DiffusIE/config.ini',
                        help="The path to config file of system")
    parser.add_argument("--output_dir", type=str, default='/home/daclai/DiffusECI/experiments',
                        help="The path to save model checkpoints and logs")
    parser.add_argument("--hf_cache", type=str, default='/home/daclai/DiffusECI/hf_cache',
                        help="The path to save huggingface cache")
    parser.add_argument("--cache", type=str, default='/home/daclai/DiffusECI/cache',
                        help="The path to save system cache")
    parser.add_argument('--devices', nargs='+', type=int,
                        help="Specify which gpu will use")
    parser.add_argument('--tuning', action='store_true', default=False, 
                        help='Tune hyperparameters or not')
    parser.add_argument('--training', action='store_true', default=False, 
                        help='Traning the model')
    parser.add_argument('--testing', action='store_true', default=False, 
                        help='Runing the test experiment')
    parser.add_argument("--load_checkpoint", type=str, default=None,
                        help="The path to checkpoint of system")
    
    return parser.parse_args()



