binary_learning_map = {
    'Cell' : 1,
    'Cell-Multi' : 1,
    'Cracking' : 1, 
    'Hot-Spot' : 1, 
    'Hot-Spot-Multi' : 1, 
    'Shadowing' : 1,
    'Diode' : 1, 
    'Diode-Multi' : 1, 
    'Vegetation' : 1, 
    'Soiling' : 1,
    'Offline-Module' : 1, 
    'No-Anomaly' : 0
    }

binary_inv_learning_map = {
    0 : 'No-Anomaly',
    1 : 'Anomaly'
    }

multi_learning_map = {
    'Cell' : 0,
    'Cell-Multi' : 1,
    'Cracking' : 2,
    'Hot-Spot' : 3,
    'Hot-Spot-Multi' : 4,
    'Shadowing' : 5,
    'Diode' : 6,
    'Diode-Multi' : 7,
    'Vegetation' : 8,
    'Soiling' : 9,
    'Offline-Module' : 10
    }

multi_inv_learning_map = {
    0: 'Cell',
    1: 'Cell_Multi',
    2 : 'Cracking',
    3 : 'Hot-Spot',
    4 : 'Hot-Spot-Multi',
    5 : 'Shadowing',
    6 : 'Diode',
    7 : 'Diode-Multi',
    8 : 'Vegetation',
    9 : 'Soiling',
    10 : 'Offline-Module',
    }

multi_reduction_learning_map = {
    'Cell' : 0,
    'Cell-Multi' : 1,
    'Hot-Spot' : 2,
    'Hot-Spot-Multi' : 3,
    'Diode' : 4,
    'Diode-Multi' : 5,
    'Offline-Module' : 6
    }

multi_reduction_inv_learning_map = {
    0 : 'Cell',
    1 : 'Cell_Multi',
    2 : 'Hot-Spot',
    3 : 'Hot-Spot-Multi',
    4 : 'Diode',
    5 : 'Diode-Multi',
    6 : 'Offline-Module',
    }

multi_reduction_learning_map1 = {
    'Cell' : 0,
    'Cell-Multi' : 1,
    'Diode' : 2,
    'Diode-Multi' : 3,
    'Offline-Module' : 4
    }

multi_reduction_inv_learning_map1 = {
    0 : 'Cell',
    1 : 'Cell_Multi',
    2 : 'Diode',
    3 : 'Diode-Multi',
    4 : 'Offline-Module',
    }

# binary_inv_learning_map:
#   - 0 : 'No-Anomaly'
#   - 1 : 'Anomaly'

# multi_learning_map:
#   - 'Cell' : 0
#   - 'Cell_Multi' : 1
#   - 'Cracking' : 2
#   - 'Hot-Spot' : 3
#   - 'Hot-Spot-Multi' : 4
#   - 'Shadowing' : 5
#   - 'Diode' : 6
#   - 'Diode-Multi' : 7
#   - 'Vegetation' : 8
#   - 'Soiling' : 9
#   - 'Offline-Module' : 10

# multi_inv_learning_map:
#   - 0: 'Cell'
#   - 1: 'Cell_Multi'
#   - 2 : 'Cracking'
#   - 3 : 'Hot-Spot'
#   - 4 : 'Hot-Spot-Multi'
#   - 5 : 'Shadowing'
#   - 6 : 'Diode'
#   - 7 : 'Diode-Multi' 
#   - 8 : 'Vegetation'
#   - 9 : 'Soiling'
#   - 10 : 'Offline-Module'



