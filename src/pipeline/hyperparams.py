hypers = {

    'print_freq': 100,
    'save_freq': 100,
    'batch_size': 64,
    'epochs': 400,

    # Optimization
    'learning_rate': 0.05,
    'lr_decay_epochs': "100,200,300",
    'lr_decay_rate': 0.1,
    'weight_decay': 1e-4,
    'momentum': 0.9,

    # Dataset
    'dataset': 'ffpp',
    'temp': 0.07,

    #method
    'method': 'SupCon',
    'model': 'MultiMode',

    'cosine': True,
    'warm': True,
    'trial': '0',
}