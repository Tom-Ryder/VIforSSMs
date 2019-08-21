from AR_dat_gen import data_gen
from AR import *
import sys, argparse
import numpy as np

def handle_opts():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='',
        usage='%(prog)s hyperparameters.txt [OPTIONS] \n Any options passed will be prioritised over the setting in the hyperparameters text file. \n To repair your hyperparameters file use -repair and copy and paste the output into hyperparameters.txt'
    )
    parser.add_argument('file', action='store',
                        help='File containing all hyperparameters')
    parser.add_argument('-T', '-time', action='store', dest='T', default=None, help='Time')
    parser.add_argument('-i', '-impute', action='store', dest='impute', default=None, help='Impute')
    parser.add_argument('-t', '-theta', action='append', dest='theta', default=None, help='Theta values listed')
    parser.add_argument('-x', '-xzero', action='store', dest='x0', default=None, help='Value for x at time 0')
    parser.add_argument('-o', '-obs_std', action='store', dest='obs_std', default=None, help='Observation standard deviation')
    parser.add_argument('-k', '-kernel_len', action='store', dest='kernel_len', default=None, help='Length of Kernel')
    parser.add_argument('-b', '-batch_dims', action='store', dest='batch_dims', default=None, help='Batch Dimensions')
    parser.add_argument('-f', '-feat_window', action='store', dest='feat_window', default=None, help='Feature Window')
    parser.add_argument('-repair', action='store_true', dest='repair', default=False, help='Output default hyperparameters to repair file')
    args = parser.parse_args()
    return args

def parseparams(file):
    f = open(file, "r")
    lines = f.readlines()
    params = []
    params.append(int(lines[1].rstrip()))
    params.append(int(lines[3].rstrip()))
    params.append(float(lines[5].rstrip()))
    tstring = lines[7].rstrip().split(",")
    tlist = []
    for t in tstring:
        tlist.append(float(t))
    params.append(tlist)
    params.append(float(lines[9].rstrip()))
    params.append(int(lines[11].rstrip()))
    params.append(int(lines[13].rstrip()))
    params.append(int(lines[15].rstrip()))
    dimstring = lines[17].rstrip().split(",")
    dimlist = []
    for dim in dimstring:
        dimlist.append(int(dim))
    params.append(dimlist)
    params.append(int(lines[19].rstrip()))
    tuples = lines[21].rstrip().replace(')','').split("(")
    tuples = tuples[1:]
    tuplist = []
    for tup in tuples:
        tuplist.append((float(tup.split(",")[0]),float(tup.split(",")[1])))
    params.append(tuplist)
    params.append(int(lines[23].rstrip()))
    params.append(float(lines[25].rstrip()))
    params.append(float(lines[27].rstrip()))
    return params

if __name__ == "__main__":
    args = handle_opts()
    if args.repair:
        print("""
#### T ####
5000
#### impute ####
1
#### x0 ####
10.0
#### Theta ####
5.0, 0.5, 3.0
#### Observation Standard Deviation ####
1.
#### p ####
50
#### kernel_len ####
50
#### batch_dims ####
50
#### network_dims ####
50, 50, 50
#### no_flows ####
3
####  priors ####
(0., 10.0)(0., 10.0)(0., 10.0)
#### feat_window ####
10
#### learn_rate ####
1e-3
#### grad_clip ####
2.5e8
        """)
        sys.exit("Copy the above into a .txt file")
    if args.file:
        try:
            hyperlist = parseparams(file=args.file)
            T = hyperlist[0]
            impute = hyperlist[1]
            x0 = hyperlist[2]
            theta = np.array(hyperlist[3])
            obs_std = hyperlist[4]
            p = hyperlist[5]
            kernel_len = hyperlist[6]
            batch_dims = hyperlist[7]
            network_dims = hyperlist[8]
            no_flows = hyperlist[9]
            priors = hyperlist[10]
            feat_window = hyperlist[11]
            learn_rate = hyperlist[12]
            grad_clip = hyperlist[13]
        except:
            sys.exit("Please specify a valid hyperparameter file")
    if args.T is not None:
        T = int(args.T)
    if args.impute is not None:
        impute = int(args.impute)
    if args.theta is not None:
        theta = args.theta
        theta = np.array(theta)
    if args.x0 is not None:
        x0 = float(args.x0)
    if args.obs_std is not None:
        obs_std = float(args.obs_std)
    if args.kernel_len is not None:
        kernel_len = int(args.kernel_len)
    if args.batch_dims is not None:
        batch_dims = int(args.batch_dims)
    if args.feat_window is not None:
        feat_window = int(args.feat_window)
    data_gen(T, impute, x0, theta, obs_std)
    main(p, kernel_len, T, batch_dims, network_dims, no_flows, priors, feat_window, x0, obs_std, learn_rate=learn_rate, grad_clip=grad_clip)
