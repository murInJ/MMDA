from models.framework.MMDA import MMDA

def get_model(model_name, args=None):
    model_dict = {
        "MMDA":get_MMDA,
    }
    return model_dict[model_name](args)



def get_MMDA(args):
    return MMDA()

