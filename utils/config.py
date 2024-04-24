import torch

config = {
    # Tokenizer Settings
    'lang_to_model': { "de": "de_core_news_sm", "en": "en_core_web_sm"},
    'SpecialTokens' : {
        'PAD_TOKEN': "<PAD>",
        'BOS_TOKEN': "<BOS>",
        'EOS_TOKEN': "<EOS>",
        'UNK_TOKEN': "<UNK>"
    },

    # Device Settings
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),

    # Model Hyper Parameters Settings

    # Baseline Model
    'BASELINE_MODEL_CONFIG' : {
        "num_layers": 6,
        "d_model": 512,
        "num_heads": 8,
        "dropout_prob": 0.1,
        "label_smoothing_value": 0.1,
        "max_length": 400
    },

    # Big Model
    'BIG_MODEL_CONFIG' : {
        "num_layers": 6,
        "d_model": 1024,
        "num_heads": 16,
        "dropout_prob": 0.3,
        "label_smoothing_value": 0.1,
        "max_length": 400
    },

    # Optimizer Hyper Parameters Settings
    'HyperParemeters_Base_recommended': {
        'batch_size': 4500,
        'warmup': 14_000,
    },
    'HyperParemeters_Big': {
        'batch_size': 1500,
        'warmup': 14_000,
    }
}