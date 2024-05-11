from transformers import Trainer
import torch


def param_stats(model, training=True, print_trainable=False, skip_cls=False):
    """
    Do a param count and optionally print the trainable layers
    """
    param_count = 0
    param_trainable = 0
    model_size = 0
    
    for name, param in model.named_parameters():
        param_count += torch.numel(param) 
        model_size += torch.numel(param) * param.element_size()
        
        if param.requires_grad:
            if "classifier" in name:
                if skip_cls:
                    continue 
            
            param_trainable += torch.numel(param) 
            if print_trainable:
                print(name, f": {torch.numel(param) / 1024 ** 2:.4f}M, {param.shape}")            
                
            
    # print("Total GPU memory: %.2f GB" % (torch.cuda.mem_get_info()[1] / 1024 ** 3))
    # print("Avail GPU memory %.2f GB" % (torch.cuda.mem_get_info()[0] / 1024 ** 3))
    print(
        f"Total parameters: {param_count / 1024 ** 2:.2f}M,\n \
        trainable parameters: {param_trainable / 1024 ** 2:.2f}M ({100 * param_trainable / param_count:.2f}%)\n \
        model size: {model_size / 1024 ** 2:.2f}MB"
    )
    
    if training:
        assert param_trainable != 0, "There's a bug in your code, your're training nothing!"


class MyAwesomeTrainer(Trainer):
    """
    Modified for initializing the monarch params and adding them to the optimizer
    before the 1st training step.
    """
    
    def __init__(self, *args, **kwargs):
        self.log_param_steps = kwargs.pop("log_param_steps", 800)
        self.train_step = 0
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        if self.train_step % self.log_param_steps == 0:
            param_stats(model, training=False, print_trainable=False, skip_cls=True)
        self.train_step += 1
        
        return super().training_step(model, inputs)