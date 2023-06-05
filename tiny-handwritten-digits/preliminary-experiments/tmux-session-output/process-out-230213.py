import re
import matplotlib as mpl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

train_20_test_20_rep_0 ="""Rep 0, Training with 20 data, Training at Epoch 1, train acc 0.4, train cost 1.3736, test acc 0.22, test cost 1.4097, avg epoch time 492.7483, total time 492.7483
Rep 0, Training with 20 data, Training at Epoch 2, train acc 0.5, train cost 1.3453, test acc 0.39, test cost 1.3613, avg epoch time 283.72, total time 567.4401
Rep 0, Training with 20 data, Training at Epoch 3, train acc 0.45, train cost 1.3465, test acc 0.39, test cost 1.3569, avg epoch time 214.1925, total time 642.5775
Rep 0, Training with 20 data, Training at Epoch 4, train acc 0.4, train cost 1.3595, test acc 0.34, test cost 1.3832, avg epoch time 179.3141, total time 717.2566
Rep 0, Training with 20 data, Training at Epoch 5, train acc 0.3, train cost 1.3794, test acc 0.26, test cost 1.3901, avg epoch time 158.3987, total time 791.9933
Rep 0, Training with 20 data, Training at Epoch 6, train acc 0.35, train cost 1.3631, test acc 0.22, test cost 1.3743, avg epoch time 144.6079, total time 867.6475
Rep 0, Training with 20 data, Training at Epoch 7, train acc 0.35, train cost 1.3516, test acc 0.23, test cost 1.3719, avg epoch time 134.6478, total time 942.5348
Rep 0, Training with 20 data, Training at Epoch 8, train acc 0.25, train cost 1.3593, test acc 0.19, test cost 1.3694, avg epoch time 127.1879, total time 1017.5033
Rep 0, Training with 20 data, Training at Epoch 9, train acc 0.3, train cost 1.3863, test acc 0.25, test cost 1.3764, avg epoch time 121.4456, total time 1093.0107
Rep 0, Training with 20 data, Training at Epoch 10, train acc 0.25, train cost 1.3854, test acc 0.34, test cost 1.3664, avg epoch time 116.8588, total time 1168.5879
Rep 0, Training with 20 data, Training at Epoch 11, train acc 0.35, train cost 1.3924, test acc 0.49, test cost 1.3593, avg epoch time 113.0359, total time 1243.3952
Rep 0, Training with 20 data, Training at Epoch 12, train acc 0.35, train cost 1.3783, test acc 0.51, test cost 1.3522, avg epoch time 109.8646, total time 1318.3751
Rep 0, Training with 20 data, Training at Epoch 13, train acc 0.3, train cost 1.3779, test acc 0.53, test cost 1.3524, avg epoch time 107.1773, total time 1393.3045
Rep 0, Training with 20 data, Training at Epoch 14, train acc 0.2, train cost 1.3775, test acc 0.22, test cost 1.376, avg epoch time 104.878, total time 1468.2923
Rep 0, Training with 20 data, Training at Epoch 15, train acc 0.25, train cost 1.3743, test acc 0.33, test cost 1.3691, avg epoch time 102.8718, total time 1543.0776
Rep 0, Training with 20 data, Training at Epoch 16, train acc 0.4, train cost 1.3747, test acc 0.35, test cost 1.3701, avg epoch time 101.1252, total time 1618.0037
Rep 0, Training with 20 data, Training at Epoch 17, train acc 0.2, train cost 1.3572, test acc 0.37, test cost 1.3729, avg epoch time 99.5978, total time 1693.1619
Rep 0, Training with 20 data, Training at Epoch 18, train acc 0.35, train cost 1.3661, test acc 0.32, test cost 1.3716, avg epoch time 98.2253, total time 1768.0557
Rep 0, Training with 20 data, Training at Epoch 19, train acc 0.4, train cost 1.3627, test acc 0.37, test cost 1.3704, avg epoch time 96.9905, total time 1842.8192
Rep 0, Training with 20 data, Training at Epoch 20, train acc 0.35, train cost 1.3641, test acc 0.46, test cost 1.3636, avg epoch time 95.939, total time 1918.7807
Rep 0, Training with 20 data, Training at Epoch 21, train acc 0.55, train cost 1.3727, test acc 0.46, test cost 1.3654, avg epoch time 94.954, total time 1994.0343
Rep 0, Training with 20 data, Training at Epoch 22, train acc 0.4, train cost 1.3641, test acc 0.37, test cost 1.3719, avg epoch time 94.0329, total time 2068.7248
Rep 0, Training with 20 data, Training at Epoch 23, train acc 0.4, train cost 1.3614, test acc 0.39, test cost 1.3591, avg epoch time 93.2101, total time 2143.833
Rep 0, Training with 20 data, Training at Epoch 24, train acc 0.25, train cost 1.3637, test acc 0.46, test cost 1.3543, avg epoch time 92.4743, total time 2219.3838
Rep 0, Training with 20 data, Training at Epoch 25, train acc 0.4, train cost 1.3596, test acc 0.4, test cost 1.3587, avg epoch time 91.7882, total time 2294.7061
Rep 0, Training with 20 data, Training at Epoch 26, train acc 0.3, train cost 1.3624, test acc 0.41, test cost 1.3606, avg epoch time 91.1226, total time 2369.1872
Rep 0, Training with 20 data, Training at Epoch 27, train acc 0.4, train cost 1.3496, test acc 0.35, test cost 1.3658, avg epoch time 90.5442, total time 2444.6929
Rep 0, Training with 20 data, Training at Epoch 28, train acc 0.35, train cost 1.3476, test acc 0.39, test cost 1.3629, avg epoch time 89.9922, total time 2519.7807
Rep 0, Training with 20 data, Training at Epoch 29, train acc 0.4, train cost 1.3514, test acc 0.36, test cost 1.3638, avg epoch time 89.4902, total time 2595.2145
Rep 0, Training with 20 data, Training at Epoch 30, train acc 0.4, train cost 1.3619, test acc 0.28, test cost 1.3758, avg epoch time 89.0217, total time 2670.65
Rep 0, Training with 20 data, Training at Epoch 31, train acc 0.55, train cost 1.3358, test acc 0.56, test cost 1.3417, avg epoch time 88.5729, total time 2745.7594
Rep 0, Training with 20 data, Training at Epoch 32, train acc 0.45, train cost 1.3376, test acc 0.52, test cost 1.3551, avg epoch time 88.1528, total time 2820.89
Rep 0, Training with 20 data, Training at Epoch 33, train acc 0.3, train cost 1.3556, test acc 0.33, test cost 1.3629, avg epoch time 87.7527, total time 2895.8383
Rep 0, Training with 20 data, Training at Epoch 34, train acc 0.6, train cost 1.3577, test acc 0.4, test cost 1.3534, avg epoch time 87.3762, total time 2970.7922
Rep 0, Training with 20 data, Training at Epoch 35, train acc 0.45, train cost 1.3579, test acc 0.36, test cost 1.3593, avg epoch time 87.0399, total time 3046.3976
Rep 0, Training with 20 data, Training at Epoch 36, train acc 0.65, train cost 1.3466, test acc 0.32, test cost 1.3584, avg epoch time 86.7335, total time 3122.4052
Rep 0, Training with 20 data, Training at Epoch 37, train acc 0.45, train cost 1.3513, test acc 0.31, test cost 1.3611, avg epoch time 86.4421, total time 3198.3594
Rep 0, Training with 20 data, Training at Epoch 38, train acc 0.5, train cost 1.3531, test acc 0.41, test cost 1.3611, avg epoch time 86.135, total time 3273.1284
Rep 0, Training with 20 data, Training at Epoch 39, train acc 0.4, train cost 1.3526, test acc 0.34, test cost 1.3595, avg epoch time 85.8402, total time 3347.7696
Rep 0, Training with 20 data, Training at Epoch 40, train acc 0.35, train cost 1.3585, test acc 0.33, test cost 1.3632, avg epoch time 85.57, total time 3422.798
Rep 0, Training with 20 data, Training at Epoch 41, train acc 0.4, train cost 1.3622, test acc 0.37, test cost 1.3628, avg epoch time 85.3194, total time 3498.0956
Rep 0, Training with 20 data, Training at Epoch 42, train acc 0.35, train cost 1.3727, test acc 0.38, test cost 1.3594, avg epoch time 85.0895, total time 3573.7587
Rep 0, Training with 20 data, Training at Epoch 43, train acc 0.4, train cost 1.358, test acc 0.33, test cost 1.3586, avg epoch time 84.8793, total time 3649.812
Rep 0, Training with 20 data, Training at Epoch 44, train acc 0.4, train cost 1.3609, test acc 0.37, test cost 1.3627, avg epoch time 84.6507, total time 3724.631
Rep 0, Training with 20 data, Training at Epoch 45, train acc 0.5, train cost 1.3583, test acc 0.34, test cost 1.3588, avg epoch time 84.4316, total time 3799.4235
Rep 0, Training with 20 data, Training at Epoch 46, train acc 0.35, train cost 1.3613, test acc 0.3, test cost 1.3675, avg epoch time 84.2288, total time 3874.5256
Rep 0, Training with 20 data, Training at Epoch 47, train acc 0.45, train cost 1.3608, test acc 0.29, test cost 1.3741, avg epoch time 84.0298, total time 3949.4016
Rep 0, Training with 20 data, Training at Epoch 48, train acc 0.55, train cost 1.3504, test acc 0.48, test cost 1.3575, avg epoch time 83.8645, total time 4025.4961
Rep 0, Training with 20 data, Training at Epoch 49, train acc 0.55, train cost 1.3501, test acc 0.46, test cost 1.3612, avg epoch time 83.6738, total time 4100.0155
Rep 0, Training with 20 data, Training at Epoch 50, train acc 0.5, train cost 1.3358, test acc 0.54, test cost 1.3414, avg epoch time 83.5044, total time 4175.2197
Rep 0, Training with 20 data, Training at Epoch 51, train acc 0.5, train cost 1.3365, test acc 0.54, test cost 1.3409, avg epoch time 83.3451, total time 4250.6021
Rep 0, Training with 20 data, Training at Epoch 52, train acc 0.55, train cost 1.3498, test acc 0.49, test cost 1.359, avg epoch time 83.1854, total time 4325.6393
Rep 0, Training with 20 data, Training at Epoch 53, train acc 0.45, train cost 1.3489, test acc 0.46, test cost 1.36, avg epoch time 83.0315, total time 4400.6673
Rep 0, Training with 20 data, Training at Epoch 54, train acc 0.5, train cost 1.3496, test acc 0.51, test cost 1.3554, avg epoch time 82.8876, total time 4475.932
Rep 0, Training with 20 data, Training at Epoch 55, train acc 0.4, train cost 1.3483, test acc 0.42, test cost 1.3578, avg epoch time 82.745, total time 4550.9729
Rep 0, Training with 20 data, Training at Epoch 56, train acc 0.4, train cost 1.347, test acc 0.47, test cost 1.3569, avg epoch time 82.614, total time 4626.3839
Rep 0, Training with 20 data, Training at Epoch 57, train acc 0.6, train cost 1.3555, test acc 0.42, test cost 1.3627, avg epoch time 82.4859, total time 4701.6984
Rep 0, Training with 20 data, Training at Epoch 58, train acc 0.35, train cost 1.3498, test acc 0.41, test cost 1.3634, avg epoch time 82.3525, total time 4776.4447
Rep 0, Training with 20 data, Training at Epoch 59, train acc 0.6, train cost 1.3589, test acc 0.36, test cost 1.3694, avg epoch time 82.2356, total time 4851.9007
Rep 0, Training with 20 data, Training at Epoch 60, train acc 0.65, train cost 1.3554, test acc 0.26, test cost 1.375, avg epoch time 82.1158, total time 4926.9492
Rep 0, Training with 20 data, Training at Epoch 61, train acc 0.6, train cost 1.3443, test acc 0.34, test cost 1.3733, avg epoch time 81.9992, total time 5001.9518
Rep 0, Training with 20 data, Training at Epoch 62, train acc 0.5, train cost 1.341, test acc 0.28, test cost 1.3737, avg epoch time 81.8792, total time 5076.5077
Rep 0, Training with 20 data, Training at Epoch 63, train acc 0.55, train cost 1.3501, test acc 0.27, test cost 1.3759, avg epoch time 81.7675, total time 5151.354
Rep 0, Training with 20 data, Training at Epoch 64, train acc 0.6, train cost 1.3594, test acc 0.27, test cost 1.3745, avg epoch time 81.6855, total time 5227.8717
Rep 0, Training with 20 data, Training at Epoch 65, train acc 0.5, train cost 1.3475, test acc 0.38, test cost 1.3703, avg epoch time 81.5892, total time 5303.3011
Rep 0, Training with 20 data, Training at Epoch 66, train acc 0.45, train cost 1.3578, test acc 0.39, test cost 1.3674, avg epoch time 81.4934, total time 5378.5675
Rep 0, Training with 20 data, Training at Epoch 67, train acc 0.45, train cost 1.3611, test acc 0.43, test cost 1.3697, avg epoch time 81.3885, total time 5453.0314
Rep 0, Training with 20 data, Training at Epoch 68, train acc 0.6, train cost 1.3533, test acc 0.42, test cost 1.3673, avg epoch time 81.2971, total time 5528.2046
Rep 0, Training with 20 data, Training at Epoch 69, train acc 0.4, train cost 1.3584, test acc 0.43, test cost 1.3677, avg epoch time 81.2211, total time 5604.2547
Rep 0, Training with 20 data, Training at Epoch 70, train acc 0.6, train cost 1.3473, test acc 0.43, test cost 1.375, avg epoch time 81.1417, total time 5679.9196
Rep 0, Training with 20 data, Training at Epoch 71, train acc 0.55, train cost 1.3635, test acc 0.43, test cost 1.3727, avg epoch time 81.0512, total time 5754.6324
Rep 0, Training with 20 data, Training at Epoch 72, train acc 0.65, train cost 1.3529, test acc 0.43, test cost 1.368, avg epoch time 80.9717, total time 5829.9625
Rep 0, Training with 20 data, Training at Epoch 73, train acc 0.5, train cost 1.3549, test acc 0.42, test cost 1.3678, avg epoch time 80.903, total time 5905.9188
Rep 0, Training with 20 data, Training at Epoch 74, train acc 0.6, train cost 1.3447, test acc 0.43, test cost 1.3666, avg epoch time 80.8256, total time 5981.0975
Rep 0, Training with 20 data, Training at Epoch 75, train acc 0.55, train cost 1.3474, test acc 0.35, test cost 1.3712, avg epoch time 80.7592, total time 6056.9436
Rep 0, Training with 20 data, Training at Epoch 76, train acc 0.5, train cost 1.359, test acc 0.51, test cost 1.3538, avg epoch time 80.6961, total time 6132.8998
Rep 0, Training with 20 data, Training at Epoch 77, train acc 0.45, train cost 1.3447, test acc 0.41, test cost 1.3668, avg epoch time 80.6329, total time 6208.7296
Rep 0, Training with 20 data, Training at Epoch 78, train acc 0.55, train cost 1.3477, test acc 0.38, test cost 1.3683, avg epoch time 80.5777, total time 6285.0616
Rep 0, Training with 20 data, Training at Epoch 79, train acc 0.55, train cost 1.3533, test acc 0.44, test cost 1.3658, avg epoch time 80.5198, total time 6361.0637
Rep 0, Training with 20 data, Training at Epoch 80, train acc 0.55, train cost 1.3453, test acc 0.34, test cost 1.3613, avg epoch time 80.4516, total time 6436.1273
Rep 0, Training with 20 data, Training at Epoch 81, train acc 0.6, train cost 1.3436, test acc 0.39, test cost 1.3574, avg epoch time 80.389, total time 6511.5127
Rep 0, Training with 20 data, Training at Epoch 82, train acc 0.65, train cost 1.3335, test acc 0.43, test cost 1.3597, avg epoch time 80.3274, total time 6586.8445
Rep 0, Training with 20 data, Training at Epoch 83, train acc 0.45, train cost 1.3316, test acc 0.49, test cost 1.359, avg epoch time 80.2687, total time 6662.3031
Rep 0, Training with 20 data, Training at Epoch 84, train acc 0.65, train cost 1.3353, test acc 0.46, test cost 1.3608, avg epoch time 80.2123, total time 6737.8333
Rep 0, Training with 20 data, Training at Epoch 85, train acc 0.65, train cost 1.3478, test acc 0.54, test cost 1.3547, avg epoch time 80.1586, total time 6813.4772
Rep 0, Training with 20 data, Training at Epoch 86, train acc 0.6, train cost 1.3368, test acc 0.46, test cost 1.3636, avg epoch time 80.0981, total time 6888.437
Rep 0, Training with 20 data, Training at Epoch 87, train acc 0.5, train cost 1.3548, test acc 0.38, test cost 1.3724, avg epoch time 80.0353, total time 6963.0711
Rep 0, Training with 20 data, Training at Epoch 88, train acc 0.5, train cost 1.3523, test acc 0.35, test cost 1.3736, avg epoch time 79.9862, total time 7038.7852
Rep 0, Training with 20 data, Training at Epoch 89, train acc 0.35, train cost 1.3446, test acc 0.25, test cost 1.3734, avg epoch time 79.9362, total time 7114.3188
Rep 0, Training with 20 data, Training at Epoch 90, train acc 0.45, train cost 1.3454, test acc 0.33, test cost 1.3736, avg epoch time 79.8798, total time 7189.1841
Rep 0, Training with 20 data, Training at Epoch 91, train acc 0.5, train cost 1.3376, test acc 0.33, test cost 1.3733, avg epoch time 79.8265, total time 7264.2148
Rep 0, Training with 20 data, Training at Epoch 92, train acc 0.55, train cost 1.3391, test acc 0.38, test cost 1.3701, avg epoch time 79.7814, total time 7339.8919
Rep 0, Training with 20 data, Training at Epoch 93, train acc 0.5, train cost 1.3507, test acc 0.38, test cost 1.3715, avg epoch time 79.7282, total time 7414.723
Rep 0, Training with 20 data, Training at Epoch 94, train acc 0.45, train cost 1.3468, test acc 0.41, test cost 1.3713, avg epoch time 79.6799, total time 7489.9105
Rep 0, Training with 20 data, Training at Epoch 95, train acc 0.65, train cost 1.3438, test acc 0.43, test cost 1.3671, avg epoch time 79.6388, total time 7565.6898
Rep 0, Training with 20 data, Training at Epoch 96, train acc 0.7, train cost 1.3372, test acc 0.42, test cost 1.3749, avg epoch time 79.594, total time 7641.028
Rep 0, Training with 20 data, Training at Epoch 97, train acc 0.65, train cost 1.3351, test acc 0.39, test cost 1.3743, avg epoch time 79.5494, total time 7716.2911
Rep 0, Training with 20 data, Training at Epoch 98, train acc 0.55, train cost 1.3438, test acc 0.42, test cost 1.3741, avg epoch time 79.5075, total time 7791.734
Rep 0, Training with 20 data, Training at Epoch 99, train acc 0.65, train cost 1.3458, test acc 0.4, test cost 1.376, avg epoch time 79.4609, total time 7866.6265
Rep 0, Training with 20 data, Training at Epoch 100, train acc 0.5, train cost 1.3391, test acc 0.45, test cost 1.373, avg epoch time 79.4147, total time 7941.4665
"""

train_20_test_20_rep_1 = """Rep 1, Training with 20 data, Training at Epoch 1, train acc 0.3, train cost 1.3821, test acc 0.31, test cost 1.3971, avg epoch time 497.0994, total time 497.0994
Rep 1, Training with 20 data, Training at Epoch 2, train acc 0.25, train cost 1.3756, test acc 0.27, test cost 1.384, avg epoch time 286.15, total time 572.3
Rep 1, Training with 20 data, Training at Epoch 3, train acc 0.25, train cost 1.3735, test acc 0.22, test cost 1.3861, avg epoch time 215.9846, total time 647.9538
Rep 1, Training with 20 data, Training at Epoch 4, train acc 0.3, train cost 1.3719, test acc 0.27, test cost 1.3834, avg epoch time 180.8346, total time 723.3384
Rep 1, Training with 20 data, Training at Epoch 5, train acc 0.4, train cost 1.3724, test acc 0.34, test cost 1.3763, avg epoch time 159.7354, total time 798.6769
Rep 1, Training with 20 data, Training at Epoch 6, train acc 0.35, train cost 1.3716, test acc 0.33, test cost 1.3779, avg epoch time 145.7701, total time 874.6208
Rep 1, Training with 20 data, Training at Epoch 7, train acc 0.5, train cost 1.3765, test acc 0.29, test cost 1.3811, avg epoch time 135.7103, total time 949.9718
Rep 1, Training with 20 data, Training at Epoch 8, train acc 0.4, train cost 1.3706, test acc 0.3, test cost 1.3839, avg epoch time 128.0802, total time 1024.6415
Rep 1, Training with 20 data, Training at Epoch 9, train acc 0.25, train cost 1.3655, test acc 0.26, test cost 1.3662, avg epoch time 122.1217, total time 1099.0949
Rep 1, Training with 20 data, Training at Epoch 10, train acc 0.3, train cost 1.373, test acc 0.26, test cost 1.3673, avg epoch time 117.4152, total time 1174.1525
Rep 1, Training with 20 data, Training at Epoch 11, train acc 0.3, train cost 1.3643, test acc 0.28, test cost 1.3631, avg epoch time 113.621, total time 1249.8307
Rep 1, Training with 20 data, Training at Epoch 12, train acc 0.4, train cost 1.3593, test acc 0.37, test cost 1.359, avg epoch time 110.417, total time 1325.0041
Rep 1, Training with 20 data, Training at Epoch 13, train acc 0.45, train cost 1.372, test acc 0.33, test cost 1.3635, avg epoch time 107.6915, total time 1399.9901
Rep 1, Training with 20 data, Training at Epoch 14, train acc 0.3, train cost 1.3609, test acc 0.29, test cost 1.3595, avg epoch time 105.3667, total time 1475.134
Rep 1, Training with 20 data, Training at Epoch 15, train acc 0.55, train cost 1.3485, test acc 0.38, test cost 1.3584, avg epoch time 103.3774, total time 1550.6609
Rep 1, Training with 20 data, Training at Epoch 16, train acc 0.4, train cost 1.3498, test acc 0.39, test cost 1.3588, avg epoch time 101.6541, total time 1626.4659
Rep 1, Training with 20 data, Training at Epoch 17, train acc 0.5, train cost 1.3507, test acc 0.37, test cost 1.3575, avg epoch time 100.0937, total time 1701.5935
Rep 1, Training with 20 data, Training at Epoch 18, train acc 0.5, train cost 1.3602, test acc 0.41, test cost 1.3628, avg epoch time 98.7088, total time 1776.7588
Rep 1, Training with 20 data, Training at Epoch 19, train acc 0.45, train cost 1.3464, test acc 0.37, test cost 1.3583, avg epoch time 97.4636, total time 1851.8075
Rep 1, Training with 20 data, Training at Epoch 20, train acc 0.45, train cost 1.3403, test acc 0.38, test cost 1.3569, avg epoch time 96.3729, total time 1927.4586
Rep 1, Training with 20 data, Training at Epoch 21, train acc 0.45, train cost 1.3507, test acc 0.39, test cost 1.3571, avg epoch time 95.3508, total time 2002.3676
Rep 1, Training with 20 data, Training at Epoch 22, train acc 0.4, train cost 1.3469, test acc 0.37, test cost 1.3568, avg epoch time 94.4288, total time 2077.4342
Rep 1, Training with 20 data, Training at Epoch 23, train acc 0.5, train cost 1.3337, test acc 0.37, test cost 1.3574, avg epoch time 93.6104, total time 2153.0386
Rep 1, Training with 20 data, Training at Epoch 24, train acc 0.45, train cost 1.3368, test acc 0.52, test cost 1.3466, avg epoch time 92.8503, total time 2228.4065
Rep 1, Training with 20 data, Training at Epoch 25, train acc 0.55, train cost 1.3362, test acc 0.49, test cost 1.3483, avg epoch time 92.1446, total time 2303.6159
Rep 1, Training with 20 data, Training at Epoch 26, train acc 0.5, train cost 1.3451, test acc 0.41, test cost 1.3488, avg epoch time 91.4844, total time 2378.595
Rep 1, Training with 20 data, Training at Epoch 27, train acc 0.6, train cost 1.3473, test acc 0.46, test cost 1.3525, avg epoch time 90.8666, total time 2453.3982
Rep 1, Training with 20 data, Training at Epoch 28, train acc 0.6, train cost 1.3355, test acc 0.48, test cost 1.3525, avg epoch time 90.2933, total time 2528.2111
Rep 1, Training with 20 data, Training at Epoch 29, train acc 0.6, train cost 1.3375, test acc 0.46, test cost 1.3529, avg epoch time 89.7492, total time 2602.7273
Rep 1, Training with 20 data, Training at Epoch 30, train acc 0.3, train cost 1.34, test acc 0.45, test cost 1.347, avg epoch time 89.2908, total time 2678.7254
Rep 1, Training with 20 data, Training at Epoch 31, train acc 0.55, train cost 1.3302, test acc 0.45, test cost 1.3447, avg epoch time 88.8199, total time 2753.4183
Rep 1, Training with 20 data, Training at Epoch 32, train acc 0.5, train cost 1.328, test acc 0.37, test cost 1.3498, avg epoch time 88.3864, total time 2828.3646
Rep 1, Training with 20 data, Training at Epoch 33, train acc 0.55, train cost 1.3412, test acc 0.46, test cost 1.3437, avg epoch time 88.0033, total time 2904.1087
Rep 1, Training with 20 data, Training at Epoch 34, train acc 0.45, train cost 1.3322, test acc 0.47, test cost 1.3422, avg epoch time 87.6172, total time 2978.9831
Rep 1, Training with 20 data, Training at Epoch 35, train acc 0.5, train cost 1.3263, test acc 0.46, test cost 1.3415, avg epoch time 87.2874, total time 3055.0582
Rep 1, Training with 20 data, Training at Epoch 36, train acc 0.6, train cost 1.3186, test acc 0.43, test cost 1.3412, avg epoch time 86.9861, total time 3131.4982
Rep 1, Training with 20 data, Training at Epoch 37, train acc 0.8, train cost 1.315, test acc 0.5, test cost 1.3376, avg epoch time 86.6869, total time 3207.417
Rep 1, Training with 20 data, Training at Epoch 38, train acc 0.8, train cost 1.3116, test acc 0.45, test cost 1.3309, avg epoch time 86.4192, total time 3283.9285
Rep 1, Training with 20 data, Training at Epoch 39, train acc 0.65, train cost 1.321, test acc 0.46, test cost 1.3496, avg epoch time 86.1057, total time 3358.1236
Rep 1, Training with 20 data, Training at Epoch 40, train acc 0.7, train cost 1.3207, test acc 0.43, test cost 1.3517, avg epoch time 85.8196, total time 3432.7847
Rep 1, Training with 20 data, Training at Epoch 41, train acc 0.6, train cost 1.3255, test acc 0.45, test cost 1.3496, avg epoch time 85.5529, total time 3507.6679
Rep 1, Training with 20 data, Training at Epoch 42, train acc 0.55, train cost 1.3152, test acc 0.44, test cost 1.3493, avg epoch time 85.3172, total time 3583.3229
Rep 1, Training with 20 data, Training at Epoch 43, train acc 0.55, train cost 1.3267, test acc 0.39, test cost 1.3517, avg epoch time 85.0867, total time 3658.7264
Rep 1, Training with 20 data, Training at Epoch 44, train acc 0.6, train cost 1.3191, test acc 0.49, test cost 1.3431, avg epoch time 84.8679, total time 3734.1874
Rep 1, Training with 20 data, Training at Epoch 45, train acc 0.6, train cost 1.3212, test acc 0.48, test cost 1.3407, avg epoch time 84.6606, total time 3809.7253
Rep 1, Training with 20 data, Training at Epoch 46, train acc 0.7, train cost 1.3167, test acc 0.46, test cost 1.342, avg epoch time 84.4547, total time 3884.9158
Rep 1, Training with 20 data, Training at Epoch 47, train acc 0.75, train cost 1.317, test acc 0.58, test cost 1.3363, avg epoch time 84.2417, total time 3959.3577
Rep 1, Training with 20 data, Training at Epoch 48, train acc 0.8, train cost 1.3174, test acc 0.59, test cost 1.3366, avg epoch time 84.0585, total time 4034.8098
Rep 1, Training with 20 data, Training at Epoch 49, train acc 0.7, train cost 1.3181, test acc 0.51, test cost 1.3339, avg epoch time 83.8737, total time 4109.8127
Rep 1, Training with 20 data, Training at Epoch 50, train acc 0.65, train cost 1.3215, test acc 0.51, test cost 1.3343, avg epoch time 83.7074, total time 4185.3717
Rep 1, Training with 20 data, Training at Epoch 51, train acc 0.75, train cost 1.32, test acc 0.52, test cost 1.3318, avg epoch time 83.5353, total time 4260.3017
Rep 1, Training with 20 data, Training at Epoch 52, train acc 0.75, train cost 1.3195, test acc 0.5, test cost 1.3348, avg epoch time 83.3668, total time 4335.0759
Rep 1, Training with 20 data, Training at Epoch 53, train acc 0.5, train cost 1.317, test acc 0.47, test cost 1.3367, avg epoch time 83.2153, total time 4410.4095
Rep 1, Training with 20 data, Training at Epoch 54, train acc 0.7, train cost 1.3173, test acc 0.49, test cost 1.3363, avg epoch time 83.0642, total time 4485.4641
Rep 1, Training with 20 data, Training at Epoch 55, train acc 0.5, train cost 1.3118, test acc 0.59, test cost 1.3063, avg epoch time 82.925, total time 4560.8728
Rep 1, Training with 20 data, Training at Epoch 56, train acc 0.55, train cost 1.3015, test acc 0.58, test cost 1.3034, avg epoch time 82.7991, total time 4636.7494
Rep 1, Training with 20 data, Training at Epoch 57, train acc 0.6, train cost 1.3076, test acc 0.56, test cost 1.3087, avg epoch time 82.6639, total time 4711.8438
Rep 1, Training with 20 data, Training at Epoch 58, train acc 0.6, train cost 1.2936, test acc 0.63, test cost 1.3075, avg epoch time 82.5355, total time 4787.0564
Rep 1, Training with 20 data, Training at Epoch 59, train acc 0.55, train cost 1.3149, test acc 0.66, test cost 1.3033, avg epoch time 82.4025, total time 4861.7456
Rep 1, Training with 20 data, Training at Epoch 60, train acc 0.5, train cost 1.3051, test acc 0.61, test cost 1.3073, avg epoch time 82.2814, total time 4936.8849
Rep 1, Training with 20 data, Training at Epoch 61, train acc 0.55, train cost 1.3115, test acc 0.63, test cost 1.3059, avg epoch time 82.167, total time 5012.1847
Rep 1, Training with 20 data, Training at Epoch 62, train acc 0.55, train cost 1.3122, test acc 0.66, test cost 1.3039, avg epoch time 82.0511, total time 5087.1682
Rep 1, Training with 20 data, Training at Epoch 63, train acc 0.55, train cost 1.2985, test acc 0.61, test cost 1.3046, avg epoch time 81.9404, total time 5162.2447
Rep 1, Training with 20 data, Training at Epoch 64, train acc 0.55, train cost 1.3057, test acc 0.59, test cost 1.2999, avg epoch time 81.8476, total time 5238.2441
Rep 1, Training with 20 data, Training at Epoch 65, train acc 0.5, train cost 1.3073, test acc 0.6, test cost 1.301, avg epoch time 81.7431, total time 5313.2995
Rep 1, Training with 20 data, Training at Epoch 66, train acc 0.45, train cost 1.3, test acc 0.59, test cost 1.2958, avg epoch time 81.6423, total time 5388.3886
Rep 1, Training with 20 data, Training at Epoch 67, train acc 0.55, train cost 1.2998, test acc 0.59, test cost 1.3022, avg epoch time 81.5533, total time 5464.07
Rep 1, Training with 20 data, Training at Epoch 68, train acc 0.6, train cost 1.305, test acc 0.64, test cost 1.3058, avg epoch time 81.4561, total time 5539.0159
Rep 1, Training with 20 data, Training at Epoch 69, train acc 0.55, train cost 1.3002, test acc 0.6, test cost 1.3036, avg epoch time 81.3733, total time 5614.7602
Rep 1, Training with 20 data, Training at Epoch 70, train acc 0.65, train cost 1.3011, test acc 0.63, test cost 1.3095, avg epoch time 81.2891, total time 5690.2389
Rep 1, Training with 20 data, Training at Epoch 71, train acc 0.5, train cost 1.3092, test acc 0.59, test cost 1.3013, avg epoch time 81.2028, total time 5765.402
Rep 1, Training with 20 data, Training at Epoch 72, train acc 0.55, train cost 1.3024, test acc 0.6, test cost 1.2966, avg epoch time 81.1168, total time 5840.4079
Rep 1, Training with 20 data, Training at Epoch 73, train acc 0.55, train cost 1.2981, test acc 0.54, test cost 1.3026, avg epoch time 81.0433, total time 5916.1628
Rep 1, Training with 20 data, Training at Epoch 74, train acc 0.5, train cost 1.3002, test acc 0.52, test cost 1.3014, avg epoch time 80.96, total time 5991.0377
Rep 1, Training with 20 data, Training at Epoch 75, train acc 0.55, train cost 1.3029, test acc 0.55, test cost 1.3067, avg epoch time 80.8821, total time 6066.1544
Rep 1, Training with 20 data, Training at Epoch 76, train acc 0.6, train cost 1.3033, test acc 0.51, test cost 1.3094, avg epoch time 80.8022, total time 6140.9649
Rep 1, Training with 20 data, Training at Epoch 77, train acc 0.5, train cost 1.2978, test acc 0.52, test cost 1.3129, avg epoch time 80.7415, total time 6217.0961
Rep 1, Training with 20 data, Training at Epoch 78, train acc 0.65, train cost 1.2983, test acc 0.53, test cost 1.3052, avg epoch time 80.6614, total time 6291.5918
Rep 1, Training with 20 data, Training at Epoch 79, train acc 0.6, train cost 1.2954, test acc 0.54, test cost 1.312, avg epoch time 80.5999, total time 6367.3914
Rep 1, Training with 20 data, Training at Epoch 80, train acc 0.55, train cost 1.2951, test acc 0.58, test cost 1.3082, avg epoch time 80.5253, total time 6442.0219
Rep 1, Training with 20 data, Training at Epoch 81, train acc 0.55, train cost 1.2962, test acc 0.58, test cost 1.3085, avg epoch time 80.4587, total time 6517.1527
Rep 1, Training with 20 data, Training at Epoch 82, train acc 0.6, train cost 1.2974, test acc 0.55, test cost 1.3082, avg epoch time 80.39, total time 6591.978
Rep 1, Training with 20 data, Training at Epoch 83, train acc 0.75, train cost 1.2982, test acc 0.59, test cost 1.3083, avg epoch time 80.321, total time 6666.6409
Rep 1, Training with 20 data, Training at Epoch 84, train acc 0.65, train cost 1.2962, test acc 0.54, test cost 1.3113, avg epoch time 80.266, total time 6742.3424
Rep 1, Training with 20 data, Training at Epoch 85, train acc 0.65, train cost 1.3002, test acc 0.52, test cost 1.3079, avg epoch time 80.1985, total time 6816.8747
Rep 1, Training with 20 data, Training at Epoch 86, train acc 0.7, train cost 1.2948, test acc 0.53, test cost 1.3107, avg epoch time 80.1424, total time 6892.2425
Rep 1, Training with 20 data, Training at Epoch 87, train acc 0.6, train cost 1.2968, test acc 0.57, test cost 1.3062, avg epoch time 80.0946, total time 6968.2324
Rep 1, Training with 20 data, Training at Epoch 88, train acc 0.55, train cost 1.2914, test acc 0.54, test cost 1.2999, avg epoch time 80.0429, total time 7043.7772
Rep 1, Training with 20 data, Training at Epoch 89, train acc 0.6, train cost 1.2941, test acc 0.51, test cost 1.304, avg epoch time 80.0, total time 7119.9981
Rep 1, Training with 20 data, Training at Epoch 90, train acc 0.6, train cost 1.2931, test acc 0.57, test cost 1.3041, avg epoch time 79.9385, total time 7194.4664
Rep 1, Training with 20 data, Training at Epoch 91, train acc 0.7, train cost 1.2954, test acc 0.56, test cost 1.307, avg epoch time 79.8837, total time 7269.4163
Rep 1, Training with 20 data, Training at Epoch 92, train acc 0.65, train cost 1.2906, test acc 0.51, test cost 1.3068, avg epoch time 79.8304, total time 7344.3943
Rep 1, Training with 20 data, Training at Epoch 93, train acc 0.7, train cost 1.2859, test acc 0.57, test cost 1.3011, avg epoch time 79.7825, total time 7419.771
Rep 1, Training with 20 data, Training at Epoch 94, train acc 0.6, train cost 1.2916, test acc 0.55, test cost 1.3045, avg epoch time 79.7362, total time 7495.1992
Rep 1, Training with 20 data, Training at Epoch 95, train acc 0.5, train cost 1.2834, test acc 0.5, test cost 1.3055, avg epoch time 79.6789, total time 7569.4955
Rep 1, Training with 20 data, Training at Epoch 96, train acc 0.65, train cost 1.2864, test acc 0.55, test cost 1.3011, avg epoch time 79.6393, total time 7645.3711
Rep 1, Training with 20 data, Training at Epoch 97, train acc 0.65, train cost 1.2864, test acc 0.5, test cost 1.3075, avg epoch time 79.6055, total time 7721.7332
Rep 1, Training with 20 data, Training at Epoch 98, train acc 0.7, train cost 1.2854, test acc 0.54, test cost 1.3147, avg epoch time 79.5687, total time 7797.7332
Rep 1, Training with 20 data, Training at Epoch 99, train acc 0.65, train cost 1.2859, test acc 0.51, test cost 1.3181, avg epoch time 79.5212, total time 7872.6011
Rep 1, Training with 20 data, Training at Epoch 100, train acc 0.65, train cost 1.2808, test acc 0.5, test cost 1.3199, avg epoch time 79.4711, total time 7947.1104"""

train_200_test_20_rep_0 = """Rep 0, Training with 200 data, Training at Epoch 1, train acc 0.185, train cost 1.3963, test acc 0.2, test cost 1.4016, avg epoch time 4470.946, total time 4470.946
Rep 0, Training with 200 data, Training at Epoch 2, train acc 0.335, train cost 1.3693, test acc 0.37, test cost 1.3638, avg epoch time 2420.4306, total time 4840.8611
Rep 0, Training with 200 data, Training at Epoch 3, train acc 0.505, train cost 1.3542, test acc 0.38, test cost 1.3749, avg epoch time 1736.9138, total time 5210.7414
Rep 0, Training with 200 data, Training at Epoch 4, train acc 0.165, train cost 1.3963, test acc 0.21, test cost 1.3892, avg epoch time 1394.6887, total time 5578.7547
Rep 0, Training with 200 data, Training at Epoch 5, train acc 0.335, train cost 1.3789, test acc 0.21, test cost 1.3902, avg epoch time 1189.9484, total time 5949.7422
Rep 0, Training with 200 data, Training at Epoch 6, train acc 0.385, train cost 1.3782, test acc 0.3, test cost 1.3906, avg epoch time 1052.9964, total time 6317.9785
Rep 0, Training with 200 data, Training at Epoch 7, train acc 0.285, train cost 1.3723, test acc 0.25, test cost 1.3787, avg epoch time 955.337, total time 6687.3591
Rep 0, Training with 200 data, Training at Epoch 8, train acc 0.24, train cost 1.3873, test acc 0.32, test cost 1.3835, avg epoch time 882.1878, total time 7057.5024
Rep 0, Training with 200 data, Training at Epoch 9, train acc 0.315, train cost 1.3809, test acc 0.29, test cost 1.3788, avg epoch time 825.3327, total time 7427.9939
Rep 0, Training with 200 data, Training at Epoch 10, train acc 0.285, train cost 1.377, test acc 0.35, test cost 1.3746, avg epoch time 779.8076, total time 7798.0758
Rep 0, Training with 200 data, Training at Epoch 11, train acc 0.27, train cost 1.3795, test acc 0.36, test cost 1.3765, avg epoch time 742.3188, total time 8165.5069
Rep 0, Training with 200 data, Training at Epoch 12, train acc 0.27, train cost 1.376, test acc 0.29, test cost 1.3824, avg epoch time 711.2662, total time 8535.1949
Rep 0, Training with 200 data, Training at Epoch 13, train acc 0.335, train cost 1.3765, test acc 0.28, test cost 1.3795, avg epoch time 685.0464, total time 8905.6027
Rep 0, Training with 200 data, Training at Epoch 14, train acc 0.355, train cost 1.373, test acc 0.31, test cost 1.3766, avg epoch time 662.592, total time 9276.2875
Rep 0, Training with 200 data, Training at Epoch 15, train acc 0.305, train cost 1.3701, test acc 0.32, test cost 1.3749, avg epoch time 643.013, total time 9645.1947
Rep 0, Training with 200 data, Training at Epoch 16, train acc 0.37, train cost 1.3691, test acc 0.35, test cost 1.3735, avg epoch time 625.9552, total time 10015.2839
Rep 0, Training with 200 data, Training at Epoch 17, train acc 0.4, train cost 1.3651, test acc 0.38, test cost 1.3699, avg epoch time 610.8163, total time 10383.8763
Rep 0, Training with 200 data, Training at Epoch 18, train acc 0.415, train cost 1.3613, test acc 0.42, test cost 1.3649, avg epoch time 597.4764, total time 10754.5746
Rep 0, Training with 200 data, Training at Epoch 19, train acc 0.405, train cost 1.3606, test acc 0.38, test cost 1.3699, avg epoch time 585.5216, total time 11124.9103
Rep 0, Training with 200 data, Training at Epoch 20, train acc 0.425, train cost 1.3623, test acc 0.39, test cost 1.3677, avg epoch time 574.7059, total time 11494.1181
Rep 0, Training with 200 data, Training at Epoch 21, train acc 0.42, train cost 1.3612, test acc 0.47, test cost 1.3675, avg epoch time 564.9168, total time 11863.2528
Rep 0, Training with 200 data, Training at Epoch 22, train acc 0.515, train cost 1.3568, test acc 0.44, test cost 1.3654, avg epoch time 556.1431, total time 12235.1474
Rep 0, Training with 200 data, Training at Epoch 23, train acc 0.5, train cost 1.359, test acc 0.4, test cost 1.3691, avg epoch time 548.0204, total time 12604.4686
Rep 0, Training with 200 data, Training at Epoch 24, train acc 0.52, train cost 1.3623, test acc 0.44, test cost 1.3671, avg epoch time 540.5968, total time 12974.3228
Rep 0, Training with 200 data, Training at Epoch 25, train acc 0.54, train cost 1.3534, test acc 0.54, test cost 1.3505, avg epoch time 533.7656, total time 13344.1388
Rep 0, Training with 200 data, Training at Epoch 26, train acc 0.475, train cost 1.3521, test acc 0.48, test cost 1.3553, avg epoch time 527.4282, total time 13713.1324
Rep 0, Training with 200 data, Training at Epoch 27, train acc 0.47, train cost 1.3527, test acc 0.5, test cost 1.3535, avg epoch time 521.5659, total time 14082.2781
Rep 0, Training with 200 data, Training at Epoch 28, train acc 0.5, train cost 1.355, test acc 0.5, test cost 1.3531, avg epoch time 516.1425, total time 14451.9905
Rep 0, Training with 200 data, Training at Epoch 29, train acc 0.49, train cost 1.3534, test acc 0.47, test cost 1.3574, avg epoch time 511.0848, total time 14821.4602
Rep 0, Training with 200 data, Training at Epoch 30, train acc 0.45, train cost 1.3538, test acc 0.49, test cost 1.358, avg epoch time 506.3193, total time 15189.5778
Rep 0, Training with 200 data, Training at Epoch 31, train acc 0.5, train cost 1.3514, test acc 0.45, test cost 1.3593, avg epoch time 501.9597, total time 15560.7513
Rep 0, Training with 200 data, Training at Epoch 32, train acc 0.46, train cost 1.3469, test acc 0.53, test cost 1.3521, avg epoch time 497.7919, total time 15929.3411
Rep 0, Training with 200 data, Training at Epoch 33, train acc 0.475, train cost 1.348, test acc 0.44, test cost 1.353, avg epoch time 493.8899, total time 16298.3677
Rep 0, Training with 200 data, Training at Epoch 34, train acc 0.42, train cost 1.3575, test acc 0.38, test cost 1.3672, avg epoch time 490.2575, total time 16668.7556
Rep 0, Training with 200 data, Training at Epoch 35, train acc 0.49, train cost 1.3566, test acc 0.37, test cost 1.366, avg epoch time 486.7894, total time 17037.6283
Rep 0, Training with 200 data, Training at Epoch 36, train acc 0.53, train cost 1.3541, test acc 0.46, test cost 1.3613, avg epoch time 483.521, total time 17406.7546
Rep 0, Training with 200 data, Training at Epoch 37, train acc 0.525, train cost 1.3534, test acc 0.41, test cost 1.3626, avg epoch time 480.4523, total time 17776.7356
Rep 0, Training with 200 data, Training at Epoch 38, train acc 0.52, train cost 1.3528, test acc 0.44, test cost 1.3621, avg epoch time 477.5692, total time 18147.6301
Rep 0, Training with 200 data, Training at Epoch 39, train acc 0.535, train cost 1.3542, test acc 0.47, test cost 1.3591, avg epoch time 474.7887, total time 18516.7597
Rep 0, Training with 200 data, Training at Epoch 40, train acc 0.46, train cost 1.3605, test acc 0.51, test cost 1.355, avg epoch time 472.1845, total time 18887.3818
Rep 0, Training with 200 data, Training at Epoch 41, train acc 0.525, train cost 1.3536, test acc 0.56, test cost 1.3487, avg epoch time 469.634, total time 19254.9954
Rep 0, Training with 200 data, Training at Epoch 42, train acc 0.555, train cost 1.3502, test acc 0.6, test cost 1.3454, avg epoch time 467.2433, total time 19624.2203
Rep 0, Training with 200 data, Training at Epoch 43, train acc 0.58, train cost 1.3495, test acc 0.62, test cost 1.3525, avg epoch time 464.9395, total time 19992.3967
Rep 0, Training with 200 data, Training at Epoch 44, train acc 0.57, train cost 1.3472, test acc 0.49, test cost 1.354, avg epoch time 462.7571, total time 20361.3143
Rep 0, Training with 200 data, Training at Epoch 45, train acc 0.585, train cost 1.3458, test acc 0.56, test cost 1.3537, avg epoch time 460.7557, total time 20734.0047
Rep 0, Training with 200 data, Training at Epoch 46, train acc 0.455, train cost 1.3469, test acc 0.41, test cost 1.3533, avg epoch time 458.7739, total time 21103.5984
Rep 0, Training with 200 data, Training at Epoch 47, train acc 0.38, train cost 1.3439, test acc 0.39, test cost 1.3481, avg epoch time 456.9014, total time 21474.3637
Rep 0, Training with 200 data, Training at Epoch 48, train acc 0.4, train cost 1.3445, test acc 0.43, test cost 1.3497, avg epoch time 455.069, total time 21843.3107
Rep 0, Training with 200 data, Training at Epoch 49, train acc 0.45, train cost 1.3435, test acc 0.45, test cost 1.344, avg epoch time 453.3452, total time 22213.9138
Rep 0, Training with 200 data, Training at Epoch 50, train acc 0.44, train cost 1.3413, test acc 0.43, test cost 1.3511, avg epoch time 451.6518, total time 22582.5893
Rep 0, Training with 200 data, Training at Epoch 51, train acc 0.395, train cost 1.3444, test acc 0.44, test cost 1.3464, avg epoch time 450.0527, total time 22952.6855
Rep 0, Training with 200 data, Training at Epoch 52, train acc 0.42, train cost 1.3446, test acc 0.41, test cost 1.345, avg epoch time 448.5261, total time 23323.3596
Rep 0, Training with 200 data, Training at Epoch 53, train acc 0.435, train cost 1.3406, test acc 0.4, test cost 1.3457, avg epoch time 447.0529, total time 23693.803
Rep 0, Training with 200 data, Training at Epoch 54, train acc 0.42, train cost 1.3427, test acc 0.4, test cost 1.3458, avg epoch time 445.6327, total time 24064.1648
Rep 0, Training with 200 data, Training at Epoch 55, train acc 0.42, train cost 1.3425, test acc 0.43, test cost 1.3466, avg epoch time 444.2561, total time 24434.0867
Rep 0, Training with 200 data, Training at Epoch 56, train acc 0.41, train cost 1.3425, test acc 0.48, test cost 1.3406, avg epoch time 442.946, total time 24804.9762
Rep 0, Training with 200 data, Training at Epoch 57, train acc 0.545, train cost 1.3414, test acc 0.5, test cost 1.3474, avg epoch time 441.6695, total time 25175.1594
Rep 0, Training with 200 data, Training at Epoch 58, train acc 0.62, train cost 1.3428, test acc 0.51, test cost 1.3417, avg epoch time 440.4217, total time 25544.4596
Rep 0, Training with 200 data, Training at Epoch 59, train acc 0.63, train cost 1.3345, test acc 0.51, test cost 1.3453, avg epoch time 439.2635, total time 25916.5494
Rep 0, Training with 200 data, Training at Epoch 60, train acc 0.585, train cost 1.3388, test acc 0.55, test cost 1.3472, avg epoch time 438.0825, total time 26284.9499
Rep 0, Training with 200 data, Training at Epoch 61, train acc 0.625, train cost 1.338, test acc 0.5, test cost 1.3447, avg epoch time 436.9861, total time 26656.1541
Rep 0, Training with 200 data, Training at Epoch 62, train acc 0.615, train cost 1.3377, test acc 0.52, test cost 1.346, avg epoch time 435.8972, total time 27025.6293
Rep 0, Training with 200 data, Training at Epoch 63, train acc 0.67, train cost 1.3341, test acc 0.54, test cost 1.3465, avg epoch time 434.8336, total time 27394.519
Rep 0, Training with 200 data, Training at Epoch 64, train acc 0.62, train cost 1.3339, test acc 0.49, test cost 1.3459, avg epoch time 433.8359, total time 27765.4966
Rep 0, Training with 200 data, Training at Epoch 65, train acc 0.655, train cost 1.3342, test acc 0.55, test cost 1.3457, avg epoch time 432.8537, total time 28135.4891
Rep 0, Training with 200 data, Training at Epoch 66, train acc 0.61, train cost 1.3284, test acc 0.53, test cost 1.3416, avg epoch time 431.9097, total time 28506.0428
Rep 0, Training with 200 data, Training at Epoch 67, train acc 0.565, train cost 1.3286, test acc 0.43, test cost 1.3445, avg epoch time 431.0071, total time 28877.4729
Rep 0, Training with 200 data, Training at Epoch 68, train acc 0.59, train cost 1.3333, test acc 0.39, test cost 1.3463, avg epoch time 430.1025, total time 29246.9677
Rep 0, Training with 200 data, Training at Epoch 69, train acc 0.56, train cost 1.3292, test acc 0.46, test cost 1.3436, avg epoch time 429.2447, total time 29617.8819
Rep 0, Training with 200 data, Training at Epoch 70, train acc 0.59, train cost 1.329, test acc 0.49, test cost 1.3406, avg epoch time 428.4293, total time 29990.0502
Rep 0, Training with 200 data, Training at Epoch 71, train acc 0.56, train cost 1.3271, test acc 0.51, test cost 1.3379, avg epoch time 427.6014, total time 30359.6971
Rep 0, Training with 200 data, Training at Epoch 72, train acc 0.615, train cost 1.3302, test acc 0.5, test cost 1.3392, avg epoch time 426.793, total time 30729.0983
Rep 0, Training with 200 data, Training at Epoch 73, train acc 0.6, train cost 1.3278, test acc 0.43, test cost 1.3413, avg epoch time 426.0122, total time 31098.8942
Rep 0, Training with 200 data, Training at Epoch 74, train acc 0.635, train cost 1.3274, test acc 0.48, test cost 1.3359, avg epoch time 425.2611, total time 31469.3237
Rep 0, Training with 200 data, Training at Epoch 75, train acc 0.635, train cost 1.3331, test acc 0.48, test cost 1.3394, avg epoch time 424.5622, total time 31842.1642
Rep 0, Training with 200 data, Training at Epoch 76, train acc 0.645, train cost 1.3307, test acc 0.53, test cost 1.3387, avg epoch time 423.8545, total time 32212.94
Rep 0, Training with 200 data, Training at Epoch 77, train acc 0.61, train cost 1.3308, test acc 0.55, test cost 1.3372, avg epoch time 423.1375, total time 32581.5874
Rep 0, Training with 200 data, Training at Epoch 78, train acc 0.615, train cost 1.3314, test acc 0.53, test cost 1.3369, avg epoch time 422.4473, total time 32950.8863
Rep 0, Training with 200 data, Training at Epoch 79, train acc 0.605, train cost 1.3313, test acc 0.55, test cost 1.3329, avg epoch time 421.7697, total time 33319.8045
Rep 0, Training with 200 data, Training at Epoch 80, train acc 0.575, train cost 1.3306, test acc 0.52, test cost 1.3342, avg epoch time 421.1206, total time 33689.644
Rep 0, Training with 200 data, Training at Epoch 81, train acc 0.595, train cost 1.3316, test acc 0.49, test cost 1.3367, avg epoch time 420.4654, total time 34057.6988
Rep 0, Training with 200 data, Training at Epoch 82, train acc 0.66, train cost 1.3214, test acc 0.55, test cost 1.3357, avg epoch time 419.8594, total time 34428.4728
Rep 0, Training with 200 data, Training at Epoch 83, train acc 0.65, train cost 1.324, test acc 0.56, test cost 1.3311, avg epoch time 419.2783, total time 34800.1001
Rep 0, Training with 200 data, Training at Epoch 84, train acc 0.655, train cost 1.3228, test acc 0.53, test cost 1.3339, avg epoch time 418.6824, total time 35169.323
Rep 0, Training with 200 data, Training at Epoch 85, train acc 0.685, train cost 1.321, test acc 0.53, test cost 1.332, avg epoch time 418.0881, total time 35537.4847
Rep 0, Training with 200 data, Training at Epoch 86, train acc 0.655, train cost 1.3217, test acc 0.46, test cost 1.3344, avg epoch time 417.5198, total time 35906.7053
Rep 0, Training with 200 data, Training at Epoch 87, train acc 0.615, train cost 1.3209, test acc 0.49, test cost 1.3371, avg epoch time 416.9569, total time 36275.2533
Rep 0, Training with 200 data, Training at Epoch 88, train acc 0.705, train cost 1.3161, test acc 0.58, test cost 1.3275, avg epoch time 416.4146, total time 36644.4839
Rep 0, Training with 200 data, Training at Epoch 89, train acc 0.63, train cost 1.3169, test acc 0.48, test cost 1.3313, avg epoch time 415.9047, total time 37015.5211
Rep 0, Training with 200 data, Training at Epoch 90, train acc 0.625, train cost 1.3184, test acc 0.49, test cost 1.3336, avg epoch time 415.3808, total time 37384.2751
Rep 0, Training with 200 data, Training at Epoch 91, train acc 0.695, train cost 1.3228, test acc 0.58, test cost 1.3362, avg epoch time 414.8941, total time 37755.3598
Rep 0, Training with 200 data, Training at Epoch 92, train acc 0.675, train cost 1.3208, test acc 0.58, test cost 1.3315, avg epoch time 414.3971, total time 38124.5291
Rep 0, Training with 200 data, Training at Epoch 93, train acc 0.66, train cost 1.3149, test acc 0.56, test cost 1.326, avg epoch time 413.9183, total time 38494.4062
Rep 0, Training with 200 data, Training at Epoch 94, train acc 0.67, train cost 1.3189, test acc 0.61, test cost 1.3271, avg epoch time 413.465, total time 38865.7119
Rep 0, Training with 200 data, Training at Epoch 95, train acc 0.68, train cost 1.3126, test acc 0.61, test cost 1.3209, avg epoch time 413.0025, total time 39235.2407
Rep 0, Training with 200 data, Training at Epoch 96, train acc 0.64, train cost 1.3186, test acc 0.6, test cost 1.3274, avg epoch time 412.567, total time 39606.4275
Rep 0, Training with 200 data, Training at Epoch 97, train acc 0.56, train cost 1.3155, test acc 0.55, test cost 1.3276, avg epoch time 412.1386, total time 39977.4421
Rep 0, Training with 200 data, Training at Epoch 98, train acc 0.57, train cost 1.3146, test acc 0.49, test cost 1.3339, avg epoch time 411.6915, total time 40345.7664
Rep 0, Training with 200 data, Training at Epoch 99, train acc 0.555, train cost 1.3148, test acc 0.52, test cost 1.329, avg epoch time 411.2837, total time 40717.085
Rep 0, Training with 200 data, Training at Epoch 100, train acc 0.555, train cost 1.3154, test acc 0.5, test cost 1.3336, avg epoch time 410.8746, total time 41087.4598"""

train_200_test_20_rep_1 = """Rep 1, Training with 200 data, Training at Epoch 1, train acc 0.205, train cost 1.379, test acc 0.28, test cost 1.3794, avg epoch time 4479.7921, total time 4479.7921
Rep 1, Training with 200 data, Training at Epoch 2, train acc 0.18, train cost 1.3785, test acc 0.31, test cost 1.373, avg epoch time 2425.3091, total time 4850.6181
Rep 1, Training with 200 data, Training at Epoch 3, train acc 0.08, train cost 1.4071, test acc 0.11, test cost 1.4103, avg epoch time 1741.0703, total time 5223.2108
Rep 1, Training with 200 data, Training at Epoch 4, train acc 0.37, train cost 1.3714, test acc 0.39, test cost 1.3767, avg epoch time 1398.3561, total time 5593.4245
Rep 1, Training with 200 data, Training at Epoch 5, train acc 0.215, train cost 1.3859, test acc 0.26, test cost 1.3882, avg epoch time 1192.4677, total time 5962.3384
Rep 1, Training with 200 data, Training at Epoch 6, train acc 0.255, train cost 1.3876, test acc 0.14, test cost 1.3874, avg epoch time 1055.3878, total time 6332.3268
Rep 1, Training with 200 data, Training at Epoch 7, train acc 0.22, train cost 1.387, test acc 0.17, test cost 1.3881, avg epoch time 957.5764, total time 6703.0347
Rep 1, Training with 200 data, Training at Epoch 8, train acc 0.265, train cost 1.3861, test acc 0.29, test cost 1.3807, avg epoch time 883.9461, total time 7071.5692
Rep 1, Training with 200 data, Training at Epoch 9, train acc 0.25, train cost 1.3819, test acc 0.26, test cost 1.3767, avg epoch time 826.8225, total time 7441.4026
Rep 1, Training with 200 data, Training at Epoch 10, train acc 0.32, train cost 1.381, test acc 0.39, test cost 1.3748, avg epoch time 781.2347, total time 7812.3467
Rep 1, Training with 200 data, Training at Epoch 11, train acc 0.315, train cost 1.3808, test acc 0.34, test cost 1.3718, avg epoch time 743.8058, total time 8181.864
Rep 1, Training with 200 data, Training at Epoch 12, train acc 0.335, train cost 1.3771, test acc 0.29, test cost 1.3741, avg epoch time 712.6302, total time 8551.5623
Rep 1, Training with 200 data, Training at Epoch 13, train acc 0.325, train cost 1.3745, test acc 0.37, test cost 1.3731, avg epoch time 686.3259, total time 8922.2373
Rep 1, Training with 200 data, Training at Epoch 14, train acc 0.33, train cost 1.3769, test acc 0.35, test cost 1.3753, avg epoch time 663.7953, total time 9293.1337
Rep 1, Training with 200 data, Training at Epoch 15, train acc 0.3, train cost 1.3773, test acc 0.45, test cost 1.3718, avg epoch time 644.1445, total time 9662.1675
Rep 1, Training with 200 data, Training at Epoch 16, train acc 0.4, train cost 1.367, test acc 0.42, test cost 1.3725, avg epoch time 627.0013, total time 10032.0208
Rep 1, Training with 200 data, Training at Epoch 17, train acc 0.47, train cost 1.3634, test acc 0.5, test cost 1.3636, avg epoch time 611.8189, total time 10400.922
Rep 1, Training with 200 data, Training at Epoch 18, train acc 0.48, train cost 1.3572, test acc 0.49, test cost 1.3663, avg epoch time 598.2913, total time 10769.2437
Rep 1, Training with 200 data, Training at Epoch 19, train acc 0.475, train cost 1.3586, test acc 0.52, test cost 1.3599, avg epoch time 586.2819, total time 11139.3559
Rep 1, Training with 200 data, Training at Epoch 20, train acc 0.44, train cost 1.3577, test acc 0.4, test cost 1.3646, avg epoch time 575.3995, total time 11507.9893
Rep 1, Training with 200 data, Training at Epoch 21, train acc 0.455, train cost 1.3551, test acc 0.41, test cost 1.3621, avg epoch time 565.6249, total time 11878.1229
Rep 1, Training with 200 data, Training at Epoch 22, train acc 0.51, train cost 1.3585, test acc 0.36, test cost 1.3652, avg epoch time 556.7371, total time 12248.2169
Rep 1, Training with 200 data, Training at Epoch 23, train acc 0.39, train cost 1.3703, test acc 0.27, test cost 1.377, avg epoch time 548.7163, total time 12620.475
Rep 1, Training with 200 data, Training at Epoch 24, train acc 0.415, train cost 1.3593, test acc 0.41, test cost 1.3602, avg epoch time 541.3107, total time 12991.457
Rep 1, Training with 200 data, Training at Epoch 25, train acc 0.485, train cost 1.3583, test acc 0.45, test cost 1.362, avg epoch time 534.4245, total time 13360.6133
Rep 1, Training with 200 data, Training at Epoch 26, train acc 0.48, train cost 1.3599, test acc 0.44, test cost 1.3609, avg epoch time 528.1117, total time 13730.9036
Rep 1, Training with 200 data, Training at Epoch 27, train acc 0.48, train cost 1.3609, test acc 0.46, test cost 1.3616, avg epoch time 522.2019, total time 14099.45
Rep 1, Training with 200 data, Training at Epoch 28, train acc 0.48, train cost 1.3568, test acc 0.41, test cost 1.3617, avg epoch time 516.7492, total time 14468.9778
Rep 1, Training with 200 data, Training at Epoch 29, train acc 0.535, train cost 1.3529, test acc 0.42, test cost 1.3647, avg epoch time 511.6804, total time 14838.7323
Rep 1, Training with 200 data, Training at Epoch 30, train acc 0.465, train cost 1.3543, test acc 0.39, test cost 1.361, avg epoch time 506.9912, total time 15209.7368
Rep 1, Training with 200 data, Training at Epoch 31, train acc 0.495, train cost 1.3542, test acc 0.41, test cost 1.3581, avg epoch time 502.5797, total time 15579.9708
Rep 1, Training with 200 data, Training at Epoch 32, train acc 0.505, train cost 1.3557, test acc 0.38, test cost 1.3593, avg epoch time 498.467, total time 15950.9456
Rep 1, Training with 200 data, Training at Epoch 33, train acc 0.575, train cost 1.3456, test acc 0.62, test cost 1.3483, avg epoch time 494.6337, total time 16322.9134
Rep 1, Training with 200 data, Training at Epoch 34, train acc 0.585, train cost 1.3486, test acc 0.56, test cost 1.3497, avg epoch time 490.9882, total time 16693.5979
Rep 1, Training with 200 data, Training at Epoch 35, train acc 0.61, train cost 1.3479, test acc 0.57, test cost 1.3466, avg epoch time 487.5122, total time 17062.9278
Rep 1, Training with 200 data, Training at Epoch 36, train acc 0.44, train cost 1.3502, test acc 0.37, test cost 1.3637, avg epoch time 484.3164, total time 17435.3904
Rep 1, Training with 200 data, Training at Epoch 37, train acc 0.46, train cost 1.3515, test acc 0.4, test cost 1.3641, avg epoch time 481.2816, total time 17807.4204
Rep 1, Training with 200 data, Training at Epoch 38, train acc 0.43, train cost 1.353, test acc 0.36, test cost 1.3632, avg epoch time 478.4421, total time 18180.798
Rep 1, Training with 200 data, Training at Epoch 39, train acc 0.465, train cost 1.3491, test acc 0.38, test cost 1.3622, avg epoch time 475.6725, total time 18551.227
Rep 1, Training with 200 data, Training at Epoch 40, train acc 0.495, train cost 1.3455, test acc 0.38, test cost 1.3633, avg epoch time 473.0584, total time 18922.3357
Rep 1, Training with 200 data, Training at Epoch 41, train acc 0.605, train cost 1.3383, test acc 0.54, test cost 1.3488, avg epoch time 470.5748, total time 19293.5684
Rep 1, Training with 200 data, Training at Epoch 42, train acc 0.655, train cost 1.339, test acc 0.49, test cost 1.3505, avg epoch time 468.154, total time 19662.4673
Rep 1, Training with 200 data, Training at Epoch 43, train acc 0.645, train cost 1.3397, test acc 0.5, test cost 1.3508, avg epoch time 465.8695, total time 20032.3877
Rep 1, Training with 200 data, Training at Epoch 44, train acc 0.55, train cost 1.34, test acc 0.47, test cost 1.351, avg epoch time 463.7239, total time 20403.8504
Rep 1, Training with 200 data, Training at Epoch 45, train acc 0.535, train cost 1.3344, test acc 0.46, test cost 1.3484, avg epoch time 461.646, total time 20774.0721
Rep 1, Training with 200 data, Training at Epoch 46, train acc 0.535, train cost 1.3375, test acc 0.43, test cost 1.3586, avg epoch time 459.7055, total time 21146.455
Rep 1, Training with 200 data, Training at Epoch 47, train acc 0.625, train cost 1.3331, test acc 0.52, test cost 1.3449, avg epoch time 457.7938, total time 21516.3081
Rep 1, Training with 200 data, Training at Epoch 48, train acc 0.53, train cost 1.3328, test acc 0.55, test cost 1.3409, avg epoch time 455.9747, total time 21886.7841
Rep 1, Training with 200 data, Training at Epoch 49, train acc 0.605, train cost 1.3339, test acc 0.54, test cost 1.3423, avg epoch time 454.2738, total time 22259.4171
Rep 1, Training with 200 data, Training at Epoch 50, train acc 0.665, train cost 1.3349, test acc 0.54, test cost 1.3488, avg epoch time 452.5975, total time 22629.8769
Rep 1, Training with 200 data, Training at Epoch 51, train acc 0.575, train cost 1.3574, test acc 0.5, test cost 1.364, avg epoch time 450.9706, total time 22999.5017
Rep 1, Training with 200 data, Training at Epoch 52, train acc 0.64, train cost 1.3487, test acc 0.53, test cost 1.3598, avg epoch time 449.4411, total time 23370.9374
Rep 1, Training with 200 data, Training at Epoch 53, train acc 0.605, train cost 1.3483, test acc 0.45, test cost 1.3594, avg epoch time 447.9393, total time 23740.7809
Rep 1, Training with 200 data, Training at Epoch 54, train acc 0.48, train cost 1.347, test acc 0.39, test cost 1.3644, avg epoch time 446.5575, total time 24114.1059
Rep 1, Training with 200 data, Training at Epoch 55, train acc 0.505, train cost 1.3464, test acc 0.4, test cost 1.3639, avg epoch time 445.1686, total time 24484.2737
Rep 1, Training with 200 data, Training at Epoch 56, train acc 0.51, train cost 1.3466, test acc 0.44, test cost 1.3594, avg epoch time 443.8202, total time 24853.9323
Rep 1, Training with 200 data, Training at Epoch 57, train acc 0.42, train cost 1.3509, test acc 0.33, test cost 1.3604, avg epoch time 442.5487, total time 25225.2737
Rep 1, Training with 200 data, Training at Epoch 58, train acc 0.415, train cost 1.3488, test acc 0.37, test cost 1.3618, avg epoch time 441.2948, total time 25595.0993
Rep 1, Training with 200 data, Training at Epoch 59, train acc 0.45, train cost 1.3514, test acc 0.35, test cost 1.3567, avg epoch time 440.1235, total time 25967.286
Rep 1, Training with 200 data, Training at Epoch 60, train acc 0.45, train cost 1.3497, test acc 0.41, test cost 1.36, avg epoch time 439.0023, total time 26340.1405
Rep 1, Training with 200 data, Training at Epoch 61, train acc 0.385, train cost 1.3501, test acc 0.41, test cost 1.359, avg epoch time 437.8947, total time 26711.574
Rep 1, Training with 200 data, Training at Epoch 62, train acc 0.445, train cost 1.3545, test acc 0.38, test cost 1.3578, avg epoch time 436.813, total time 27082.4059
Rep 1, Training with 200 data, Training at Epoch 63, train acc 0.495, train cost 1.3476, test acc 0.44, test cost 1.3528, avg epoch time 435.7929, total time 27454.9503
Rep 1, Training with 200 data, Training at Epoch 64, train acc 0.56, train cost 1.3417, test acc 0.46, test cost 1.3466, avg epoch time 434.8024, total time 27827.3527
Rep 1, Training with 200 data, Training at Epoch 65, train acc 0.52, train cost 1.3343, test acc 0.38, test cost 1.3494, avg epoch time 433.8291, total time 28198.8916
Rep 1, Training with 200 data, Training at Epoch 66, train acc 0.525, train cost 1.3361, test acc 0.48, test cost 1.3467, avg epoch time 432.8743, total time 28569.7043
Rep 1, Training with 200 data, Training at Epoch 67, train acc 0.535, train cost 1.3325, test acc 0.44, test cost 1.3451, avg epoch time 431.9594, total time 28941.2807
Rep 1, Training with 200 data, Training at Epoch 68, train acc 0.53, train cost 1.3315, test acc 0.45, test cost 1.3429, avg epoch time 431.0404, total time 29310.7484
Rep 1, Training with 200 data, Training at Epoch 69, train acc 0.535, train cost 1.3437, test acc 0.46, test cost 1.3588, avg epoch time 430.1843, total time 29682.7133
Rep 1, Training with 200 data, Training at Epoch 70, train acc 0.545, train cost 1.3428, test acc 0.46, test cost 1.3519, avg epoch time 429.3252, total time 30052.7644
Rep 1, Training with 200 data, Training at Epoch 71, train acc 0.575, train cost 1.3428, test acc 0.5, test cost 1.3554, avg epoch time 428.4911, total time 30422.8689
Rep 1, Training with 200 data, Training at Epoch 72, train acc 0.585, train cost 1.3354, test acc 0.5, test cost 1.3467, avg epoch time 427.7106, total time 30795.1663
Rep 1, Training with 200 data, Training at Epoch 73, train acc 0.56, train cost 1.3366, test acc 0.49, test cost 1.3437, avg epoch time 426.9372, total time 31166.4163
Rep 1, Training with 200 data, Training at Epoch 74, train acc 0.54, train cost 1.3377, test acc 0.49, test cost 1.3471, avg epoch time 426.1659, total time 31536.2788
Rep 1, Training with 200 data, Training at Epoch 75, train acc 0.53, train cost 1.3335, test acc 0.47, test cost 1.3468, avg epoch time 425.4287, total time 31907.1493
Rep 1, Training with 200 data, Training at Epoch 76, train acc 0.53, train cost 1.3344, test acc 0.49, test cost 1.3481, avg epoch time 424.703, total time 32277.4314
Rep 1, Training with 200 data, Training at Epoch 77, train acc 0.535, train cost 1.3367, test acc 0.48, test cost 1.3468, avg epoch time 424.0148, total time 32649.1407
Rep 1, Training with 200 data, Training at Epoch 78, train acc 0.53, train cost 1.3309, test acc 0.46, test cost 1.3469, avg epoch time 423.3663, total time 33022.569
Rep 1, Training with 200 data, Training at Epoch 79, train acc 0.53, train cost 1.3311, test acc 0.5, test cost 1.3379, avg epoch time 422.6863, total time 33392.2196
Rep 1, Training with 200 data, Training at Epoch 80, train acc 0.535, train cost 1.3314, test acc 0.51, test cost 1.3368, avg epoch time 422.0008, total time 33760.0609
Rep 1, Training with 200 data, Training at Epoch 81, train acc 0.49, train cost 1.3219, test acc 0.43, test cost 1.3511, avg epoch time 421.3918, total time 34132.7347
Rep 1, Training with 200 data, Training at Epoch 82, train acc 0.535, train cost 1.3298, test acc 0.45, test cost 1.3516, avg epoch time 420.7829, total time 34504.2008
Rep 1, Training with 200 data, Training at Epoch 83, train acc 0.53, train cost 1.3229, test acc 0.45, test cost 1.3356, avg epoch time 420.1887, total time 34875.6591
Rep 1, Training with 200 data, Training at Epoch 84, train acc 0.52, train cost 1.3204, test acc 0.48, test cost 1.3239, avg epoch time 419.5884, total time 35245.4266
Rep 1, Training with 200 data, Training at Epoch 85, train acc 0.515, train cost 1.3193, test acc 0.52, test cost 1.3261, avg epoch time 419.0217, total time 35616.8435
Rep 1, Training with 200 data, Training at Epoch 86, train acc 0.5, train cost 1.3157, test acc 0.48, test cost 1.3244, avg epoch time 418.4618, total time 35987.7156
Rep 1, Training with 200 data, Training at Epoch 87, train acc 0.485, train cost 1.3167, test acc 0.46, test cost 1.3213, avg epoch time 417.9081, total time 36358.0086
Rep 1, Training with 200 data, Training at Epoch 88, train acc 0.485, train cost 1.3153, test acc 0.46, test cost 1.319, avg epoch time 417.3736, total time 36728.8739
Rep 1, Training with 200 data, Training at Epoch 89, train acc 0.54, train cost 1.3172, test acc 0.47, test cost 1.3318, avg epoch time 416.8476, total time 37099.433
Rep 1, Training with 200 data, Training at Epoch 90, train acc 0.57, train cost 1.3206, test acc 0.51, test cost 1.3267, avg epoch time 416.3242, total time 37469.1771
Rep 1, Training with 200 data, Training at Epoch 91, train acc 0.56, train cost 1.3141, test acc 0.51, test cost 1.3311, avg epoch time 415.8031, total time 37838.0828
Rep 1, Training with 200 data, Training at Epoch 92, train acc 0.535, train cost 1.3149, test acc 0.46, test cost 1.3328, avg epoch time 415.3, total time 38207.6024
Rep 1, Training with 200 data, Training at Epoch 93, train acc 0.535, train cost 1.3137, test acc 0.49, test cost 1.3361, avg epoch time 414.8327, total time 38579.4365
Rep 1, Training with 200 data, Training at Epoch 94, train acc 0.54, train cost 1.3156, test acc 0.46, test cost 1.3284, avg epoch time 414.3626, total time 38950.081
Rep 1, Training with 200 data, Training at Epoch 95, train acc 0.555, train cost 1.3117, test acc 0.5, test cost 1.3308, avg epoch time 413.8916, total time 39319.7057
Rep 1, Training with 200 data, Training at Epoch 96, train acc 0.54, train cost 1.3166, test acc 0.48, test cost 1.3324, avg epoch time 413.4382, total time 39690.0719
Rep 1, Training with 200 data, Training at Epoch 97, train acc 0.545, train cost 1.3305, test acc 0.48, test cost 1.347, avg epoch time 413.0008, total time 40061.0811
Rep 1, Training with 200 data, Training at Epoch 98, train acc 0.51, train cost 1.3187, test acc 0.45, test cost 1.3313, avg epoch time 412.5783, total time 40432.6716
Rep 1, Training with 200 data, Training at Epoch 99, train acc 0.53, train cost 1.3185, test acc 0.5, test cost 1.3238, avg epoch time 412.1448, total time 40802.3381
Rep 1, Training with 200 data, Training at Epoch 100, train acc 0.535, train cost 1.319, test acc 0.49, test cost 1.3241, avg epoch time 411.7201, total time 41172.0127"""

train_500_test_20_rep_0 = """Rep 0, Training with 500 data, Training at Epoch 1, train acc 0.176, train cost 1.385, test acc 0.21, test cost 1.3825, avg epoch time 11143.9904, total time 11143.9904
Rep 0, Training with 500 data, Training at Epoch 2, train acc 0.21, train cost 1.3798, test acc 0.26, test cost 1.3772, avg epoch time 6006.6602, total time 12013.3203
Rep 0, Training with 500 data, Training at Epoch 3, train acc 0.204, train cost 1.3953, test acc 0.17, test cost 1.3904, avg epoch time 4294.3545, total time 12883.0636
Rep 0, Training with 500 data, Training at Epoch 4, train acc 0.214, train cost 1.3902, test acc 0.18, test cost 1.3893, avg epoch time 3436.4073, total time 13745.6293
Rep 0, Training with 500 data, Training at Epoch 5, train acc 0.24, train cost 1.388, test acc 0.19, test cost 1.3882, avg epoch time 2921.8691, total time 14609.3453
Rep 0, Training with 500 data, Training at Epoch 6, train acc 0.284, train cost 1.3906, test acc 0.28, test cost 1.3858, avg epoch time 2579.9064, total time 15479.4383
Rep 0, Training with 500 data, Training at Epoch 7, train acc 0.206, train cost 1.3916, test acc 0.18, test cost 1.3994, avg epoch time 2334.8533, total time 16343.9733
Rep 0, Training with 500 data, Training at Epoch 8, train acc 0.356, train cost 1.3794, test acc 0.32, test cost 1.3839, avg epoch time 2151.3388, total time 17210.7102
Rep 0, Training with 500 data, Training at Epoch 9, train acc 0.37, train cost 1.3785, test acc 0.27, test cost 1.3871, avg epoch time 2008.6241, total time 18077.6167
Rep 0, Training with 500 data, Training at Epoch 10, train acc 0.37, train cost 1.3773, test acc 0.32, test cost 1.3826, avg epoch time 1894.5288, total time 18945.288
Rep 0, Training with 500 data, Training at Epoch 11, train acc 0.314, train cost 1.3784, test acc 0.21, test cost 1.3838, avg epoch time 1800.8857, total time 19809.7424
Rep 0, Training with 500 data, Training at Epoch 12, train acc 0.342, train cost 1.3758, test acc 0.3, test cost 1.3829, avg epoch time 1723.1394, total time 20677.6733
Rep 0, Training with 500 data, Training at Epoch 13, train acc 0.376, train cost 1.3767, test acc 0.36, test cost 1.3775, avg epoch time 1657.4597, total time 21546.9755
Rep 0, Training with 500 data, Training at Epoch 14, train acc 0.386, train cost 1.3745, test acc 0.39, test cost 1.3779, avg epoch time 1600.8782, total time 22412.2954
Rep 0, Training with 500 data, Training at Epoch 15, train acc 0.388, train cost 1.3735, test acc 0.38, test cost 1.3738, avg epoch time 1552.521, total time 23287.8149
Rep 0, Training with 500 data, Training at Epoch 16, train acc 0.494, train cost 1.3671, test acc 0.5, test cost 1.3652, avg epoch time 1509.3503, total time 24149.6042
Rep 0, Training with 500 data, Training at Epoch 17, train acc 0.522, train cost 1.3669, test acc 0.48, test cost 1.3659, avg epoch time 1471.4475, total time 25014.6081
Rep 0, Training with 500 data, Training at Epoch 18, train acc 0.412, train cost 1.3672, test acc 0.37, test cost 1.3676, avg epoch time 1437.8679, total time 25881.6222
Rep 0, Training with 500 data, Training at Epoch 19, train acc 0.388, train cost 1.37, test acc 0.35, test cost 1.369, avg epoch time 1407.9224, total time 26750.5259
Rep 0, Training with 500 data, Training at Epoch 20, train acc 0.45, train cost 1.3668, test acc 0.47, test cost 1.368, avg epoch time 1380.6617, total time 27613.2349
Rep 0, Training with 500 data, Training at Epoch 21, train acc 0.5, train cost 1.3572, test acc 0.49, test cost 1.357, avg epoch time 1356.2007, total time 28480.2151
Rep 0, Training with 500 data, Training at Epoch 22, train acc 0.542, train cost 1.3529, test acc 0.51, test cost 1.3597, avg epoch time 1333.9924, total time 29347.8334
Rep 0, Training with 500 data, Training at Epoch 23, train acc 0.598, train cost 1.3494, test acc 0.47, test cost 1.3514, avg epoch time 1313.6807, total time 30214.6569
Rep 0, Training with 500 data, Training at Epoch 24, train acc 0.546, train cost 1.3505, test acc 0.51, test cost 1.3521, avg epoch time 1295.0866, total time 31082.0782
Rep 0, Training with 500 data, Training at Epoch 25, train acc 0.558, train cost 1.3505, test acc 0.5, test cost 1.3566, avg epoch time 1277.8811, total time 31947.0273
Rep 0, Training with 500 data, Training at Epoch 26, train acc 0.402, train cost 1.3547, test acc 0.37, test cost 1.3588, avg epoch time 1262.0528, total time 32813.3732
Rep 0, Training with 500 data, Training at Epoch 27, train acc 0.462, train cost 1.3522, test acc 0.46, test cost 1.3535, avg epoch time 1247.31, total time 33677.3687
Rep 0, Training with 500 data, Training at Epoch 28, train acc 0.444, train cost 1.3532, test acc 0.46, test cost 1.355, avg epoch time 1233.694, total time 34543.4326
Rep 0, Training with 500 data, Training at Epoch 29, train acc 0.494, train cost 1.3505, test acc 0.49, test cost 1.3528, avg epoch time 1220.9685, total time 35408.0857
Rep 0, Training with 500 data, Training at Epoch 30, train acc 0.474, train cost 1.352, test acc 0.5, test cost 1.352, avg epoch time 1209.088, total time 36272.6392
Rep 0, Training with 500 data, Training at Epoch 31, train acc 0.486, train cost 1.3509, test acc 0.48, test cost 1.3516, avg epoch time 1197.9584, total time 37136.7094
Rep 0, Training with 500 data, Training at Epoch 32, train acc 0.53, train cost 1.3505, test acc 0.56, test cost 1.349, avg epoch time 1187.6003, total time 38003.209
Rep 0, Training with 500 data, Training at Epoch 33, train acc 0.482, train cost 1.3522, test acc 0.49, test cost 1.3525, avg epoch time 1177.8034, total time 38867.5116
Rep 0, Training with 500 data, Training at Epoch 34, train acc 0.502, train cost 1.3473, test acc 0.58, test cost 1.348, avg epoch time 1168.6981, total time 39735.7359
Rep 0, Training with 500 data, Training at Epoch 35, train acc 0.432, train cost 1.3508, test acc 0.45, test cost 1.3477, avg epoch time 1159.9848, total time 40599.468
Rep 0, Training with 500 data, Training at Epoch 36, train acc 0.35, train cost 1.3472, test acc 0.42, test cost 1.3434, avg epoch time 1151.725, total time 41462.1007
Rep 0, Training with 500 data, Training at Epoch 37, train acc 0.422, train cost 1.3411, test acc 0.41, test cost 1.339, avg epoch time 1143.9969, total time 42327.8857
Rep 0, Training with 500 data, Training at Epoch 38, train acc 0.44, train cost 1.3392, test acc 0.44, test cost 1.3412, avg epoch time 1136.5398, total time 43188.5116
Rep 0, Training with 500 data, Training at Epoch 39, train acc 0.448, train cost 1.3405, test acc 0.39, test cost 1.3416, avg epoch time 1129.5225, total time 44051.3764
Rep 0, Training with 500 data, Training at Epoch 40, train acc 0.416, train cost 1.3397, test acc 0.42, test cost 1.34, avg epoch time 1122.9499, total time 44917.9954
Rep 0, Training with 500 data, Training at Epoch 41, train acc 0.442, train cost 1.359, test acc 0.4, test cost 1.3572, avg epoch time 1116.685, total time 45784.0852
Rep 0, Training with 500 data, Training at Epoch 42, train acc 0.374, train cost 1.3547, test acc 0.38, test cost 1.3533, avg epoch time 1110.6512, total time 46647.3504
Rep 0, Training with 500 data, Training at Epoch 43, train acc 0.386, train cost 1.3522, test acc 0.46, test cost 1.3514, avg epoch time 1104.8867, total time 47510.126
Rep 0, Training with 500 data, Training at Epoch 44, train acc 0.348, train cost 1.3545, test acc 0.32, test cost 1.3517, avg epoch time 1099.4267, total time 48374.7734
Rep 0, Training with 500 data, Training at Epoch 45, train acc 0.344, train cost 1.3527, test acc 0.34, test cost 1.3537, avg epoch time 1094.2155, total time 49239.699
Rep 0, Training with 500 data, Training at Epoch 46, train acc 0.346, train cost 1.3499, test acc 0.4, test cost 1.3434, avg epoch time 1089.2958, total time 50107.6048
Rep 0, Training with 500 data, Training at Epoch 47, train acc 0.348, train cost 1.3496, test acc 0.36, test cost 1.3481, avg epoch time 1084.4443, total time 50968.8825
Rep 0, Training with 500 data, Training at Epoch 48, train acc 0.46, train cost 1.3416, test acc 0.45, test cost 1.3434, avg epoch time 1079.9632, total time 51838.2352
Rep 0, Training with 500 data, Training at Epoch 49, train acc 0.454, train cost 1.3433, test acc 0.42, test cost 1.3375, avg epoch time 1075.5797, total time 52703.4058
Rep 0, Training with 500 data, Training at Epoch 50, train acc 0.576, train cost 1.3278, test acc 0.59, test cost 1.3266, avg epoch time 1071.2902, total time 53564.5102
Rep 0, Training with 500 data, Training at Epoch 51, train acc 0.478, train cost 1.3323, test acc 0.52, test cost 1.3255, avg epoch time 1067.1365, total time 54423.9625
Rep 0, Training with 500 data, Training at Epoch 52, train acc 0.536, train cost 1.3337, test acc 0.56, test cost 1.3309, avg epoch time 1063.2682, total time 55289.9463
Rep 0, Training with 500 data, Training at Epoch 53, train acc 0.556, train cost 1.3347, test acc 0.56, test cost 1.3379, avg epoch time 1059.5901, total time 56158.2744
Rep 0, Training with 500 data, Training at Epoch 54, train acc 0.582, train cost 1.3346, test acc 0.57, test cost 1.3341, avg epoch time 1055.9158, total time 57019.4535
Rep 0, Training with 500 data, Training at Epoch 55, train acc 0.55, train cost 1.331, test acc 0.61, test cost 1.3304, avg epoch time 1052.4637, total time 57885.5013
Rep 0, Training with 500 data, Training at Epoch 56, train acc 0.574, train cost 1.3299, test acc 0.56, test cost 1.3266, avg epoch time 1049.1527, total time 58752.5532
Rep 0, Training with 500 data, Training at Epoch 57, train acc 0.558, train cost 1.3297, test acc 0.55, test cost 1.3294, avg epoch time 1045.9251, total time 59617.731
Rep 0, Training with 500 data, Training at Epoch 58, train acc 0.504, train cost 1.329, test acc 0.5, test cost 1.3246, avg epoch time 1042.7932, total time 60482.0084
Rep 0, Training with 500 data, Training at Epoch 59, train acc 0.52, train cost 1.328, test acc 0.55, test cost 1.3268, avg epoch time 1039.825, total time 61349.6757
Rep 0, Training with 500 data, Training at Epoch 60, train acc 0.59, train cost 1.3238, test acc 0.58, test cost 1.3311, avg epoch time 1036.9349, total time 62216.0942
Rep 0, Training with 500 data, Training at Epoch 61, train acc 0.604, train cost 1.3214, test acc 0.56, test cost 1.3303, avg epoch time 1034.1363, total time 63082.312
Rep 0, Training with 500 data, Training at Epoch 62, train acc 0.596, train cost 1.3196, test acc 0.58, test cost 1.3303, avg epoch time 1031.354, total time 63943.9511
Rep 0, Training with 500 data, Training at Epoch 63, train acc 0.588, train cost 1.322, test acc 0.57, test cost 1.327, avg epoch time 1028.6682, total time 64806.0943
Rep 0, Training with 500 data, Training at Epoch 64, train acc 0.594, train cost 1.323, test acc 0.6, test cost 1.3269, avg epoch time 1026.1256, total time 65672.0397
Rep 0, Training with 500 data, Training at Epoch 65, train acc 0.592, train cost 1.3204, test acc 0.58, test cost 1.3304, avg epoch time 1023.6348, total time 66536.2625
Rep 0, Training with 500 data, Training at Epoch 66, train acc 0.602, train cost 1.3206, test acc 0.58, test cost 1.3258, avg epoch time 1021.2363, total time 67401.594
Rep 0, Training with 500 data, Training at Epoch 67, train acc 0.564, train cost 1.3225, test acc 0.57, test cost 1.3265, avg epoch time 1018.8908, total time 68265.6858
Rep 0, Training with 500 data, Training at Epoch 68, train acc 0.594, train cost 1.319, test acc 0.57, test cost 1.3296, avg epoch time 1016.6229, total time 69130.3547
Rep 0, Training with 500 data, Training at Epoch 69, train acc 0.582, train cost 1.3211, test acc 0.59, test cost 1.3256, avg epoch time 1014.4717, total time 69998.5454
Rep 0, Training with 500 data, Training at Epoch 70, train acc 0.554, train cost 1.3218, test acc 0.55, test cost 1.3268, avg epoch time 1012.3658, total time 70865.607
Rep 0, Training with 500 data, Training at Epoch 71, train acc 0.56, train cost 1.32, test acc 0.57, test cost 1.3238, avg epoch time 1010.2836, total time 71730.1338
Rep 0, Training with 500 data, Training at Epoch 72, train acc 0.548, train cost 1.32, test acc 0.56, test cost 1.3219, avg epoch time 1008.2855, total time 72596.5555
Rep 0, Training with 500 data, Training at Epoch 73, train acc 0.594, train cost 1.3146, test acc 0.62, test cost 1.32, avg epoch time 1006.2566, total time 73456.7287
Rep 0, Training with 500 data, Training at Epoch 74, train acc 0.584, train cost 1.3121, test acc 0.57, test cost 1.316, avg epoch time 1004.3391, total time 74321.0924
Rep 0, Training with 500 data, Training at Epoch 75, train acc 0.586, train cost 1.3108, test acc 0.62, test cost 1.316, avg epoch time 1002.4612, total time 75184.5877
Rep 0, Training with 500 data, Training at Epoch 76, train acc 0.626, train cost 1.3105, test acc 0.62, test cost 1.314, avg epoch time 1000.6074, total time 76046.1612
Rep 0, Training with 500 data, Training at Epoch 77, train acc 0.662, train cost 1.3083, test acc 0.68, test cost 1.3019, avg epoch time 998.8494, total time 76911.402
Rep 0, Training with 500 data, Training at Epoch 78, train acc 0.704, train cost 1.3065, test acc 0.72, test cost 1.305, avg epoch time 997.1692, total time 77779.1981
Rep 0, Training with 500 data, Training at Epoch 79, train acc 0.668, train cost 1.3042, test acc 0.65, test cost 1.3119, avg epoch time 995.5076, total time 78645.1009
Rep 0, Training with 500 data, Training at Epoch 80, train acc 0.654, train cost 1.3026, test acc 0.67, test cost 1.3053, avg epoch time 993.8113, total time 79504.9057
Rep 0, Training with 500 data, Training at Epoch 81, train acc 0.65, train cost 1.304, test acc 0.64, test cost 1.3056, avg epoch time 992.2507, total time 80372.3105
Rep 0, Training with 500 data, Training at Epoch 82, train acc 0.652, train cost 1.303, test acc 0.62, test cost 1.3023, avg epoch time 990.7413, total time 81240.7854
Rep 0, Training with 500 data, Training at Epoch 83, train acc 0.672, train cost 1.3014, test acc 0.69, test cost 1.3002, avg epoch time 989.2095, total time 82104.3912
Rep 0, Training with 500 data, Training at Epoch 84, train acc 0.644, train cost 1.3016, test acc 0.67, test cost 1.3018, avg epoch time 987.7451, total time 82970.5843
Rep 0, Training with 500 data, Training at Epoch 85, train acc 0.678, train cost 1.3034, test acc 0.64, test cost 1.2999, avg epoch time 986.293, total time 83834.9086
Rep 0, Training with 500 data, Training at Epoch 86, train acc 0.554, train cost 1.3136, test acc 0.48, test cost 1.3223, avg epoch time 984.8909, total time 84700.6153
Rep 0, Training with 500 data, Training at Epoch 87, train acc 0.628, train cost 1.3074, test acc 0.57, test cost 1.3115, avg epoch time 983.5212, total time 85566.345
Rep 0, Training with 500 data, Training at Epoch 88, train acc 0.6, train cost 1.3054, test acc 0.59, test cost 1.308, avg epoch time 982.1893, total time 86432.6608
Rep 0, Training with 500 data, Training at Epoch 89, train acc 0.582, train cost 1.3057, test acc 0.58, test cost 1.3121, avg epoch time 980.8978, total time 87299.9067
Rep 0, Training with 500 data, Training at Epoch 90, train acc 0.614, train cost 1.3041, test acc 0.59, test cost 1.3069, avg epoch time 979.6323, total time 88166.9076
Rep 0, Training with 500 data, Training at Epoch 91, train acc 0.556, train cost 1.306, test acc 0.46, test cost 1.3106, avg epoch time 978.4122, total time 89035.5134
Rep 0, Training with 500 data, Training at Epoch 92, train acc 0.548, train cost 1.3039, test acc 0.52, test cost 1.3125, avg epoch time 977.1937, total time 89901.8228
Rep 0, Training with 500 data, Training at Epoch 93, train acc 0.59, train cost 1.3017, test acc 0.59, test cost 1.3078, avg epoch time 975.9909, total time 90767.158
Rep 0, Training with 500 data, Training at Epoch 94, train acc 0.598, train cost 1.3018, test acc 0.63, test cost 1.3014, avg epoch time 974.7701, total time 91628.3898
Rep 0, Training with 500 data, Training at Epoch 95, train acc 0.598, train cost 1.3017, test acc 0.58, test cost 1.3057, avg epoch time 973.5826, total time 92490.3448
Rep 0, Training with 500 data, Training at Epoch 96, train acc 0.582, train cost 1.3012, test acc 0.55, test cost 1.3058, avg epoch time 972.4237, total time 93352.6774
Rep 0, Training with 500 data, Training at Epoch 97, train acc 0.574, train cost 1.2993, test acc 0.62, test cost 1.3052, avg epoch time 971.3138, total time 94217.4351
Rep 0, Training with 500 data, Training at Epoch 98, train acc 0.652, train cost 1.3019, test acc 0.67, test cost 1.3017, avg epoch time 970.237, total time 95083.2239
Rep 0, Training with 500 data, Training at Epoch 99, train acc 0.546, train cost 1.304, test acc 0.51, test cost 1.3075, avg epoch time 969.1234, total time 95943.22
Rep 0, Training with 500 data, Training at Epoch 100, train acc 0.546, train cost 1.3017, test acc 0.51, test cost 1.3057, avg epoch time 968.1082, total time 96810.824"""

train_500_test_20_rep_1 = """Rep 1, Training with 500 data, Training at Epoch 1, train acc 0.286, train cost 1.3973, test acc 0.32, test cost 1.3988, avg epoch time 11158.1426, total time 11158.1426
Rep 1, Training with 500 data, Training at Epoch 2, train acc 0.29, train cost 1.3833, test acc 0.26, test cost 1.3854, avg epoch time 6011.4365, total time 12022.873
Rep 1, Training with 500 data, Training at Epoch 3, train acc 0.204, train cost 1.392, test acc 0.15, test cost 1.3913, avg epoch time 4296.4327, total time 12889.2981
Rep 1, Training with 500 data, Training at Epoch 4, train acc 0.184, train cost 1.3925, test acc 0.13, test cost 1.3954, avg epoch time 3438.592, total time 13754.3678
Rep 1, Training with 500 data, Training at Epoch 5, train acc 0.358, train cost 1.3834, test acc 0.3, test cost 1.3821, avg epoch time 2923.897, total time 14619.485
Rep 1, Training with 500 data, Training at Epoch 6, train acc 0.352, train cost 1.3826, test acc 0.35, test cost 1.385, avg epoch time 2581.703, total time 15490.2179
Rep 1, Training with 500 data, Training at Epoch 7, train acc 0.266, train cost 1.3827, test acc 0.31, test cost 1.3805, avg epoch time 2336.5523, total time 16355.866
Rep 1, Training with 500 data, Training at Epoch 8, train acc 0.244, train cost 1.387, test acc 0.21, test cost 1.3838, avg epoch time 2152.6253, total time 17221.0024
Rep 1, Training with 500 data, Training at Epoch 9, train acc 0.316, train cost 1.3688, test acc 0.23, test cost 1.3747, avg epoch time 2009.637, total time 18086.7329
Rep 1, Training with 500 data, Training at Epoch 10, train acc 0.458, train cost 1.368, test acc 0.36, test cost 1.3737, avg epoch time 1895.18, total time 18951.8
Rep 1, Training with 500 data, Training at Epoch 11, train acc 0.372, train cost 1.3643, test acc 0.41, test cost 1.3636, avg epoch time 1801.8499, total time 19820.349
Rep 1, Training with 500 data, Training at Epoch 12, train acc 0.39, train cost 1.3662, test acc 0.44, test cost 1.3638, avg epoch time 1724.0484, total time 20688.5813
Rep 1, Training with 500 data, Training at Epoch 13, train acc 0.36, train cost 1.3654, test acc 0.43, test cost 1.3624, avg epoch time 1658.0814, total time 21555.0579
Rep 1, Training with 500 data, Training at Epoch 14, train acc 0.416, train cost 1.3613, test acc 0.43, test cost 1.3591, avg epoch time 1601.4424, total time 22420.194
Rep 1, Training with 500 data, Training at Epoch 15, train acc 0.448, train cost 1.363, test acc 0.34, test cost 1.3629, avg epoch time 1552.5256, total time 23287.8838
Rep 1, Training with 500 data, Training at Epoch 16, train acc 0.442, train cost 1.3667, test acc 0.4, test cost 1.3689, avg epoch time 1509.8728, total time 24157.9653
Rep 1, Training with 500 data, Training at Epoch 17, train acc 0.366, train cost 1.3723, test acc 0.38, test cost 1.3719, avg epoch time 1471.931, total time 25022.8274
Rep 1, Training with 500 data, Training at Epoch 18, train acc 0.276, train cost 1.3741, test acc 0.28, test cost 1.3756, avg epoch time 1438.1995, total time 25887.5904
Rep 1, Training with 500 data, Training at Epoch 19, train acc 0.44, train cost 1.3481, test acc 0.41, test cost 1.3521, avg epoch time 1408.0611, total time 26753.16
Rep 1, Training with 500 data, Training at Epoch 20, train acc 0.484, train cost 1.3528, test acc 0.45, test cost 1.3492, avg epoch time 1380.8546, total time 27617.0912
Rep 1, Training with 500 data, Training at Epoch 21, train acc 0.446, train cost 1.3508, test acc 0.43, test cost 1.3528, avg epoch time 1356.3841, total time 28484.067
Rep 1, Training with 500 data, Training at Epoch 22, train acc 0.512, train cost 1.3493, test acc 0.5, test cost 1.3515, avg epoch time 1334.1967, total time 29352.3271
Rep 1, Training with 500 data, Training at Epoch 23, train acc 0.484, train cost 1.3487, test acc 0.49, test cost 1.3483, avg epoch time 1313.7435, total time 30216.0994
Rep 1, Training with 500 data, Training at Epoch 24, train acc 0.474, train cost 1.3485, test acc 0.49, test cost 1.3501, avg epoch time 1295.0764, total time 31081.8341
Rep 1, Training with 500 data, Training at Epoch 25, train acc 0.418, train cost 1.3521, test acc 0.39, test cost 1.3554, avg epoch time 1277.9978, total time 31949.9451
Rep 1, Training with 500 data, Training at Epoch 26, train acc 0.388, train cost 1.3495, test acc 0.33, test cost 1.3563, avg epoch time 1262.039, total time 32813.0138
Rep 1, Training with 500 data, Training at Epoch 27, train acc 0.494, train cost 1.3488, test acc 0.49, test cost 1.3507, avg epoch time 1247.2332, total time 33675.2965
Rep 1, Training with 500 data, Training at Epoch 28, train acc 0.382, train cost 1.3618, test acc 0.39, test cost 1.3665, avg epoch time 1233.6587, total time 34542.4431
Rep 1, Training with 500 data, Training at Epoch 29, train acc 0.454, train cost 1.361, test acc 0.43, test cost 1.3612, avg epoch time 1221.0081, total time 35409.2345
Rep 1, Training with 500 data, Training at Epoch 30, train acc 0.44, train cost 1.3598, test acc 0.39, test cost 1.3652, avg epoch time 1209.1616, total time 36274.8479
Rep 1, Training with 500 data, Training at Epoch 31, train acc 0.446, train cost 1.3606, test acc 0.41, test cost 1.3583, avg epoch time 1198.0463, total time 37139.4359
Rep 1, Training with 500 data, Training at Epoch 32, train acc 0.462, train cost 1.3545, test acc 0.38, test cost 1.358, avg epoch time 1187.7023, total time 38006.4739
Rep 1, Training with 500 data, Training at Epoch 33, train acc 0.408, train cost 1.3543, test acc 0.41, test cost 1.3584, avg epoch time 1177.9301, total time 38871.6917
Rep 1, Training with 500 data, Training at Epoch 34, train acc 0.414, train cost 1.3561, test acc 0.46, test cost 1.3521, avg epoch time 1168.6782, total time 39735.0598
Rep 1, Training with 500 data, Training at Epoch 35, train acc 0.486, train cost 1.3534, test acc 0.42, test cost 1.3507, avg epoch time 1159.9821, total time 40599.3735
Rep 1, Training with 500 data, Training at Epoch 36, train acc 0.432, train cost 1.3517, test acc 0.5, test cost 1.3516, avg epoch time 1151.8043, total time 41464.9531
Rep 1, Training with 500 data, Training at Epoch 37, train acc 0.5, train cost 1.3491, test acc 0.5, test cost 1.3479, avg epoch time 1144.1052, total time 42331.8926
Rep 1, Training with 500 data, Training at Epoch 38, train acc 0.486, train cost 1.3481, test acc 0.53, test cost 1.3459, avg epoch time 1136.8885, total time 43201.7627
Rep 1, Training with 500 data, Training at Epoch 39, train acc 0.46, train cost 1.3474, test acc 0.46, test cost 1.349, avg epoch time 1129.861, total time 44064.5802
Rep 1, Training with 500 data, Training at Epoch 40, train acc 0.432, train cost 1.354, test acc 0.39, test cost 1.356, avg epoch time 1123.1997, total time 44927.9862
Rep 1, Training with 500 data, Training at Epoch 41, train acc 0.49, train cost 1.3562, test acc 0.52, test cost 1.3507, avg epoch time 1116.8028, total time 45788.9158
Rep 1, Training with 500 data, Training at Epoch 42, train acc 0.536, train cost 1.3526, test acc 0.48, test cost 1.3501, avg epoch time 1110.8295, total time 46654.8399
Rep 1, Training with 500 data, Training at Epoch 43, train acc 0.51, train cost 1.3517, test acc 0.55, test cost 1.3509, avg epoch time 1104.9985, total time 47514.9352
Rep 1, Training with 500 data, Training at Epoch 44, train acc 0.49, train cost 1.3527, test acc 0.49, test cost 1.3509, avg epoch time 1099.5069, total time 48378.3058
Rep 1, Training with 500 data, Training at Epoch 45, train acc 0.366, train cost 1.3651, test acc 0.33, test cost 1.3692, avg epoch time 1094.4287, total time 49249.2924
Rep 1, Training with 500 data, Training at Epoch 46, train acc 0.352, train cost 1.3648, test acc 0.35, test cost 1.3688, avg epoch time 1089.4473, total time 50114.5769
Rep 1, Training with 500 data, Training at Epoch 47, train acc 0.34, train cost 1.3654, test acc 0.32, test cost 1.3704, avg epoch time 1084.711, total time 50981.4178
Rep 1, Training with 500 data, Training at Epoch 48, train acc 0.388, train cost 1.3615, test acc 0.37, test cost 1.3605, avg epoch time 1080.156, total time 51847.4869
Rep 1, Training with 500 data, Training at Epoch 49, train acc 0.38, train cost 1.3617, test acc 0.37, test cost 1.3615, avg epoch time 1075.7937, total time 52713.89
Rep 1, Training with 500 data, Training at Epoch 50, train acc 0.524, train cost 1.3505, test acc 0.53, test cost 1.3493, avg epoch time 1071.5726, total time 53578.6308
Rep 1, Training with 500 data, Training at Epoch 51, train acc 0.55, train cost 1.3498, test acc 0.56, test cost 1.3454, avg epoch time 1067.5873, total time 54446.9522
Rep 1, Training with 500 data, Training at Epoch 52, train acc 0.54, train cost 1.3509, test acc 0.57, test cost 1.3488, avg epoch time 1063.7334, total time 55314.1393
Rep 1, Training with 500 data, Training at Epoch 53, train acc 0.55, train cost 1.3515, test acc 0.58, test cost 1.3468, avg epoch time 1059.9479, total time 56177.2398
Rep 1, Training with 500 data, Training at Epoch 54, train acc 0.56, train cost 1.3466, test acc 0.61, test cost 1.3381, avg epoch time 1056.3228, total time 57041.4316
Rep 1, Training with 500 data, Training at Epoch 55, train acc 0.596, train cost 1.3427, test acc 0.67, test cost 1.3372, avg epoch time 1052.8335, total time 57905.8424
Rep 1, Training with 500 data, Training at Epoch 56, train acc 0.616, train cost 1.3409, test acc 0.67, test cost 1.3359, avg epoch time 1049.3974, total time 58766.2526
Rep 1, Training with 500 data, Training at Epoch 57, train acc 0.484, train cost 1.3516, test acc 0.45, test cost 1.3516, avg epoch time 1046.1821, total time 59632.3823
Rep 1, Training with 500 data, Training at Epoch 58, train acc 0.486, train cost 1.3459, test acc 0.48, test cost 1.3428, avg epoch time 1043.0906, total time 60499.2556
Rep 1, Training with 500 data, Training at Epoch 59, train acc 0.492, train cost 1.3425, test acc 0.58, test cost 1.3422, avg epoch time 1040.1159, total time 61366.8393
Rep 1, Training with 500 data, Training at Epoch 60, train acc 0.506, train cost 1.3423, test acc 0.59, test cost 1.3388, avg epoch time 1037.2043, total time 62232.2592
Rep 1, Training with 500 data, Training at Epoch 61, train acc 0.54, train cost 1.3411, test acc 0.59, test cost 1.3421, avg epoch time 1034.369, total time 63096.5097
Rep 1, Training with 500 data, Training at Epoch 62, train acc 0.546, train cost 1.3409, test acc 0.62, test cost 1.3363, avg epoch time 1031.6299, total time 63961.0563
Rep 1, Training with 500 data, Training at Epoch 63, train acc 0.544, train cost 1.3413, test acc 0.57, test cost 1.3424, avg epoch time 1028.9856, total time 64826.0932
Rep 1, Training with 500 data, Training at Epoch 64, train acc 0.522, train cost 1.3417, test acc 0.58, test cost 1.3401, avg epoch time 1026.4951, total time 65695.6885
Rep 1, Training with 500 data, Training at Epoch 65, train acc 0.53, train cost 1.3408, test acc 0.57, test cost 1.3392, avg epoch time 1024.0615, total time 66563.9995
Rep 1, Training with 500 data, Training at Epoch 66, train acc 0.528, train cost 1.3464, test acc 0.53, test cost 1.3452, avg epoch time 1021.6088, total time 67426.1782
Rep 1, Training with 500 data, Training at Epoch 67, train acc 0.532, train cost 1.3442, test acc 0.57, test cost 1.3427, avg epoch time 1019.3106, total time 68293.8111
Rep 1, Training with 500 data, Training at Epoch 68, train acc 0.586, train cost 1.3458, test acc 0.54, test cost 1.3418, avg epoch time 1017.1189, total time 69164.0819
Rep 1, Training with 500 data, Training at Epoch 69, train acc 0.538, train cost 1.343, test acc 0.57, test cost 1.3352, avg epoch time 1014.8908, total time 70027.4656
Rep 1, Training with 500 data, Training at Epoch 70, train acc 0.548, train cost 1.339, test acc 0.53, test cost 1.3353, avg epoch time 1012.766, total time 70893.6198
Rep 1, Training with 500 data, Training at Epoch 71, train acc 0.528, train cost 1.3385, test acc 0.56, test cost 1.3319, avg epoch time 1010.6876, total time 71758.821
Rep 1, Training with 500 data, Training at Epoch 72, train acc 0.504, train cost 1.3378, test acc 0.53, test cost 1.3294, avg epoch time 1008.6909, total time 72625.7429
Rep 1, Training with 500 data, Training at Epoch 73, train acc 0.544, train cost 1.3367, test acc 0.54, test cost 1.3344, avg epoch time 1006.7646, total time 73493.8129
Rep 1, Training with 500 data, Training at Epoch 74, train acc 0.542, train cost 1.3378, test acc 0.53, test cost 1.3367, avg epoch time 1004.8685, total time 74360.272
Rep 1, Training with 500 data, Training at Epoch 75, train acc 0.564, train cost 1.334, test acc 0.53, test cost 1.3309, avg epoch time 1002.9398, total time 75220.4887
Rep 1, Training with 500 data, Training at Epoch 76, train acc 0.6, train cost 1.3299, test acc 0.56, test cost 1.3289, avg epoch time 1001.1653, total time 76088.5622
Rep 1, Training with 500 data, Training at Epoch 77, train acc 0.58, train cost 1.3323, test acc 0.62, test cost 1.3259, avg epoch time 999.3705, total time 76951.531
Rep 1, Training with 500 data, Training at Epoch 78, train acc 0.556, train cost 1.3323, test acc 0.57, test cost 1.3231, avg epoch time 997.697, total time 77820.3629
Rep 1, Training with 500 data, Training at Epoch 79, train acc 0.57, train cost 1.3307, test acc 0.52, test cost 1.3307, avg epoch time 996.0263, total time 78686.0774
Rep 1, Training with 500 data, Training at Epoch 80, train acc 0.576, train cost 1.3322, test acc 0.59, test cost 1.3331, avg epoch time 994.4268, total time 79554.1438
Rep 1, Training with 500 data, Training at Epoch 81, train acc 0.586, train cost 1.3329, test acc 0.59, test cost 1.3259, avg epoch time 992.8343, total time 80419.5809
Rep 1, Training with 500 data, Training at Epoch 82, train acc 0.546, train cost 1.3264, test acc 0.53, test cost 1.3282, avg epoch time 991.2541, total time 81282.8382
Rep 1, Training with 500 data, Training at Epoch 83, train acc 0.56, train cost 1.3279, test acc 0.55, test cost 1.3271, avg epoch time 989.7504, total time 82149.2793
Rep 1, Training with 500 data, Training at Epoch 84, train acc 0.542, train cost 1.3294, test acc 0.48, test cost 1.3331, avg epoch time 988.2405, total time 83012.2023
Rep 1, Training with 500 data, Training at Epoch 85, train acc 0.542, train cost 1.3304, test acc 0.5, test cost 1.3324, avg epoch time 986.7668, total time 83875.1738
Rep 1, Training with 500 data, Training at Epoch 86, train acc 0.512, train cost 1.3293, test acc 0.51, test cost 1.3316, avg epoch time 985.3562, total time 84740.6324
Rep 1, Training with 500 data, Training at Epoch 87, train acc 0.516, train cost 1.3297, test acc 0.54, test cost 1.3312, avg epoch time 983.9386, total time 85602.6625
Rep 1, Training with 500 data, Training at Epoch 88, train acc 0.466, train cost 1.3291, test acc 0.51, test cost 1.3309, avg epoch time 982.5242, total time 86462.126
Rep 1, Training with 500 data, Training at Epoch 89, train acc 0.47, train cost 1.3277, test acc 0.51, test cost 1.3246, avg epoch time 981.1984, total time 87326.6564
Rep 1, Training with 500 data, Training at Epoch 90, train acc 0.454, train cost 1.3266, test acc 0.45, test cost 1.3277, avg epoch time 979.9566, total time 88196.091
Rep 1, Training with 500 data, Training at Epoch 91, train acc 0.466, train cost 1.3262, test acc 0.54, test cost 1.3255, avg epoch time 978.7003, total time 89061.7273
Rep 1, Training with 500 data, Training at Epoch 92, train acc 0.446, train cost 1.3271, test acc 0.46, test cost 1.3273, avg epoch time 977.5022, total time 89930.206
Rep 1, Training with 500 data, Training at Epoch 93, train acc 0.47, train cost 1.3234, test acc 0.49, test cost 1.3203, avg epoch time 976.331, total time 90798.7792
Rep 1, Training with 500 data, Training at Epoch 94, train acc 0.5, train cost 1.3175, test acc 0.49, test cost 1.3155, avg epoch time 975.1964, total time 91668.4623
Rep 1, Training with 500 data, Training at Epoch 95, train acc 0.548, train cost 1.3223, test acc 0.55, test cost 1.3226, avg epoch time 974.0426, total time 92534.0469
Rep 1, Training with 500 data, Training at Epoch 96, train acc 0.59, train cost 1.3131, test acc 0.55, test cost 1.3148, avg epoch time 972.9271, total time 93401.0025
Rep 1, Training with 500 data, Training at Epoch 97, train acc 0.61, train cost 1.3167, test acc 0.61, test cost 1.3102, avg epoch time 971.8272, total time 94267.2344
Rep 1, Training with 500 data, Training at Epoch 98, train acc 0.612, train cost 1.3162, test acc 0.56, test cost 1.3147, avg epoch time 970.7471, total time 95133.2159
Rep 1, Training with 500 data, Training at Epoch 99, train acc 0.596, train cost 1.3205, test acc 0.6, test cost 1.3199, avg epoch time 969.6845, total time 95998.7681
Rep 1, Training with 500 data, Training at Epoch 100, train acc 0.616, train cost 1.3164, test acc 0.64, test cost 1.3094, avg epoch time 968.672, total time 96867.2016"""



N_EPOCHS = 100
N_TEST = 100
outputs = [train_20_test_20_rep_0, train_20_test_20_rep_1,
           train_200_test_20_rep_0, train_200_test_20_rep_1,
           train_500_test_20_rep_0, train_500_test_20_rep_1,]
train_sizes = [20, 200, 500]

def extract_acc_loss_train_test(output_text:str):
    splitted = re.split('\n',output_text)
    splitted = [c for c in splitted if len(c) > 0]
    train_acc = [float(re.search('train acc (.+?), train cost', c).group(1)) for c in splitted]
    train_loss = [float(re.search('train cost (.+?), test acc', c).group(1)) for c in splitted]
    test_acc = [float(re.search('test acc (.+?), test cost', c).group(1)) for c in splitted]
    test_loss = [float(re.search('test cost (.+?), avg epoch', c).group(1)) for c in splitted]
    n_train = [int(re.search('Training with (.+?) data, Training at Epoch', c).group(1)) for c in splitted]
    step = [int(re.search('Training at Epoch (.+?), train acc', c).group(1)) for c in splitted]
    return train_acc, train_loss, test_acc, test_loss, n_train, step

def generate_res_dataframe(output_text:str):
    train_acc, train_loss, test_acc, test_loss, n_train, step = extract_acc_loss_train_test(output_text)
    res = dict(
        n_train = n_train,
        step = step,
        train_cost = train_loss,
        train_acc = train_acc,
        test_cost = test_loss,
        test_acc = test_acc
    )
    results_df = pd.DataFrame(
        columns=["train_acc", "train_cost", "test_acc", "test_cost", "step", "n_train"]
    )
    results_df = pd.concat([results_df, pd.DataFrame.from_dict(res)], axis=0, ignore_index=True)
    return results_df

results_df = generate_res_dataframe(outputs[0])
for c in outputs[1:]:
    results_df = pd.concat([results_df, generate_res_dataframe(c)])

# aggregate dataframe
df_agg = results_df.groupby(["n_train", "step"]).agg(["mean", "std"])
df_agg = df_agg.reset_index()

sns.set_style('whitegrid')
colors = sns.color_palette()
fig, axes = plt.subplots(ncols=3, figsize=(16.5, 5))

generalization_errors = []

# plot losses and accuracies
for i, n_train in enumerate(train_sizes):
        df = df_agg[df_agg.n_train == n_train]

        dfs = [df.train_cost["mean"], df.test_cost["mean"], df.train_acc["mean"], df.test_acc["mean"]]
        lines = ["o-", "x--", "o-", "x--"]
        labels = [fr"$N={n_train}$", None, fr"$N={n_train}$", None]
        axs = [0, 0, 2, 2]

        for k in range(4):
            ax = axes[axs[k]]
            ax.plot(df.step, dfs[k], lines[k], label=labels[k], markevery=10, color=colors[i], alpha=0.8)

        # plot final loss difference
        dif = np.absolute(df[df.step == 100].test_cost["mean"] - df[df.step == 100].train_cost["mean"])
        generalization_errors.append(dif)

print(generalization_errors)

# format loss plot
ax = axes[0]
ax.set_title('Train and Test Losses', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')

# format generalization error plot
ax = axes[1]
ax.plot(train_sizes, generalization_errors, "o-", label=r"$gen(\alpha)$")
ax.set_xscale('log')
ax.set_xticks(train_sizes)
ax.set_xticklabels(train_sizes)
ax.set_title(r'Generalization Error $gen(\alpha) =Abs [R(\alpha) - \hat{R}_N(\alpha)]$', fontsize=14)
ax.set_xlabel('Training Set Size')

# format loss plot
ax = axes[2]
ax.set_title('Train and Test Accuracies', fontsize=14)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim(0., 1.05)

legend_elements = [
                          mpl.lines.Line2D([0], [0], label=f'N={n}', color=colors[i]) for i, n in enumerate(train_sizes)
                      ] + [
                          mpl.lines.Line2D([0], [0], marker='o', ls='-', label='Train', color='Black'),
                          mpl.lines.Line2D([0], [0], marker='x', ls='--', label='Test', color='Black')
                      ]

axes[0].legend(handles=legend_elements, ncol=3)
axes[2].legend(handles=legend_elements, ncol=3)

axes[1].set_yscale('log', base=2)
plt.savefig(f"qiskit-fashion-mnist-5x5-conv-multiclass-tiny-image-results-100-test-2-reps.pdf")