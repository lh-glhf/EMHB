data_collection = {
    'ETTh1': {'data_provider': 'ETTh1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
              'dataset': 'ETTh1'},
    'ETTh2': {'data_provider': 'ETTh2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
              'dataset': 'ETTh2'},
    'ETTm1': {'data_provider': 'ETTm1.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
              'dataset': 'ETTm1'},
    'ETTm2': {'data_provider': 'ETTm2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
              'dataset': 'ETTm2'},
    'weather': {'data_provider': 'weather.csv', 'T': 'OT', 'M': [21, 21, 21], 'S': [1, 1, 1], 'MS': [21, 21, 1],
                'dataset': 'custom'},
    'better_2': {'data_provider': 'better_2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},
    'better_3': {'data_provider': 'better_3.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},
    'middle_2': {'data_provider': 'middle_2.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},
    'middle_3': {'data_provider': 'middle_3.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},
    'middle_5': {'data_provider': 'middle_5.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},
    'middle_6': {'data_provider': 'middle_6.csv', 'T': 'OT', 'M': [7, 7, 7], 'S': [1, 1, 1], 'MS': [7, 7, 1],
                 'dataset': 'custom'},

}
data_split = (0.7, 0.2)  # train_ratio, test_ration if no testStamp; else train_ratio, vali_ratio
# data_testStamp = "2020/03/01 02:00:00" # the time stamp where testSet begin, set None if no need
data_testStamp = None
data_testRolling = True
data_ETT_split = False