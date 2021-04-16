config = parse_config()
tokenizer = load_tokenizer(config)
dataset = build_dataset(data_dir, tokenizer)
encoder_model = load_encoder_model(config)
