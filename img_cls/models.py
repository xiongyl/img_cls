import torchvision

model_zoo = {name: getattr(torchvision.models, name) for name in dir(torchvision.models) 
    if name.islower() and not name.startswith('_')
}
