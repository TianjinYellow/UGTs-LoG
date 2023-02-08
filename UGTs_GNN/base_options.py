import argparse


class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self,parser):

        parser.add_argument("--dataset", type=str, default="Cora", required=False,
                            help="The input dataset.",
                            choices=['Cora', 'Citeseer', 'Pubmed', 'ogbn-arxiv',
                                     'CoauthorCS', 'CoauthorPhysics', 'AmazonComputers', 'AmazonPhoto',
                                     'TEXAS', 'WISCONSIN', 'ACTOR', 'CORNELL'])
        # build up the common parameter
        parser.add_argument('--random_seed', type=int, default=None)
        parser.add_argument('--resume', action='store_true', default=False)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="run in cuda mode")


        parser.add_argument('--type_model', type=str, default="GCN",
                            choices=['GCN', 'GAT', 'SGC','GIN', 'GCNII', 'DAGNN', 'GPRGNN', 'APPNP', 'JKNet', 'DeeperGCN'])
        parser.add_argument("--dropout", type=float, default=0,help="dropout for GCN")
        #parser.add_argument('--embedding_dropout', type=float, default=0,
        #                    help='dropout for embeddings')
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument('--weight_decay', type=float, default=5e-4,
                            help="weight decay")  # 5e-4

        parser.add_argument('--transductive', type=bool, default=True,
                            help='transductive or inductive setting')
        parser.add_argument('--activation', type=str, default="relu", required=False)

        parser.add_argument('--type_norm', type=str, default="None")
        parser.add_argument('--command', type=str)
        #parser.add_argument('--config', type=str, help='file path for YAML configure file')

        #parser.add_argument('--accum_grad', type=int, default=1)
        parser.add_argument('--force_restart', action='store_true') 
        parser.add_argument('--seed_by_time',type=bool,default=True) 
        parser.add_argument('--print_train_loss',type=bool,default=False)
        parser.add_argument('--num_gpus',type=int,default=1)

        ###for config.yaml
        #parser.add_argument('--save_best_model', type=str, default=None)
        parser.add_argument('--output_dir', type=str, default='__outputs__')
        parser.add_argument('--sync_dir', type=str, default='__sync__')
        parser.add_argument('--save_best_model', type=bool, default=True)
        parser.add_argument('--optimizer', type=str, default='Adam')
        parser.add_argument('--lr_scheduler',type=str,default='MultiStepLR')
        parser.add_argument('--warm_epochs',type=int,default=0)
        parser.add_argument('--finetuning_epochs',type=int,default=0)
        parser.add_argument('--finetuning_lr',type=float,default=None)
        #parser.add_argument('--step',type=float,default=1)
        parser.add_argument('--lr_milestones',type=list,default=[900,1000,1100])
        parser.add_argument('--checkpoint_epochs',type=list,default=[])
        parser.add_argument('--multisteplr_gamma',type=float,default=0.1)
        parser.add_argument('--learning_framework',type=str,default='SupervisedLearning')
        parser.add_argument('--bn_track_running_stats',type=bool,default=True)
        parser.add_argument('--bn_affine',type=bool,default=True)

        parser.add_argument('--bn_momentum',type=float,default=0.1)

        parser.add_argument('--init_mode',type=str,default='kaiming_uniform')
        parser.add_argument('--init_mode_mask',type=str,default='kaiming_uniform')
        parser.add_argument('--init_mode_linear',type=str,default=None)
        parser.add_argument('--init_scale',type=float,default=1.0)
        parser.add_argument('--init_scale_score',type=float,default=1.0)
        parser.add_argument('--heads',type=int,default=1)

        
        #parser.add_argument('--attack',action='store_true')
        
        args = parser.parse_args()
        args = self.reset_dataset_dependent_parameters(args)
        args=self.reset_train_mode_parameters(args)

        return args

    ## setting the common hyperparameters used for comparing different methods of a trick
    def reset_train_mode_parameters(self,args):
        if args.train_mode=='normal':
           args.init_mode='kaiming_uniform'
           args.linear_sparsity=0
           #self.weight_decay=5e-4
        elif args.train_mode=='score_only':
            args.bn_affine=False
            args.type_norm='None'
            #self.weight_decay=0.0
            args.dropout=0.0
        return args



    def reset_dataset_dependent_parameters(self, args):
        if args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout = 0.6  # 0.5
            #args.lr = 0.01  # 0.005
            #args.weight_decay = 5e-4
            #args.epochs = 1000
            #args.type_norm=None
            #args.patience = 100
            #args.dim_hidden = 64
            args.activation = 'relu'

        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.dropout = 0.5
            args.lr = 0.01
            #args.weight_decay = 5e-4
            #args.epochs = 1000
            #args.patience = 100
            #args.dim_hidden = 256
            args.activation = 'relu'

        elif args.dataset == 'Citeseer':
            args.num_feats = 3703
            args.num_classes = 6

            args.dropout = 0.7
            args.lr = 0.01
            #args.lamda = 0.6
            #args.weight_decay = 5e-4
            args.activation = 'relu'
            #args.res_alpha = 0.2

        elif args.dataset == 'ogbn-arxiv':
            args.num_feats = 128
            args.num_classes = 40
            #args.lr = 0.005
            args.weight_decay = 0.
            #args.dim_hidden = 256


        # ==============================================
        return args
