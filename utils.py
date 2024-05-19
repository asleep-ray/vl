def print_number_of_trainable_model_parameters_tab(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return (
        f"{trainable_model_params:10d}\t{all_model_params:10d}\t{100 * trainable_model_params / all_model_params:.2f}%"
    )


def print_trainable_model_module(model):
    print("\t trainable\t     all\tpercentage")
    print("model", end="\t")
    print(print_number_of_trainable_model_parameters_tab(model))
    print("encoder", end="\t")
    print(print_number_of_trainable_model_parameters_tab(model.encoder))
    print("decoder", end="\t")
    print(print_number_of_trainable_model_parameters_tab(model.decoder))
    print("lm_head", end="\t")
    print(print_number_of_trainable_model_parameters_tab(model.lm_head))
