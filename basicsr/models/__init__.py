import importlib
from os import path as osp

from basicsr.utils import get_root_logger, scandir

# automatically scan and import model modules
# scan all the files under the 'models' folder and collect files ending with
# '_model.py'
model_folder = osp.dirname(osp.abspath(__file__))
model_filenames = [
    osp.splitext(osp.basename(v))[0] for v in scandir(model_folder)
    if v.endswith('_model.py')
]
# import all the model modules
_model_modules = []
for file_name in model_filenames:
    try:
        _model_modules.append(
            importlib.import_module(f'basicsr.models.{file_name}')
        )
    except Exception as e:
        print(f'WARNING: Failed to import basicsr.models.{file_name}: {e}')

# Explicit fallback: ensure ImageISBModel is always available
try:
    from basicsr.models import image_isb_model as _isb_mod
    if _isb_mod not in _model_modules:
        _model_modules.append(_isb_mod)
except Exception as e:
    print(f'WARNING: Failed to import image_isb_model: {e}')


def create_model(opt):
    """Create model.

    Args:
        opt (dict): Configuration. It constains:
            model_type (str): Model type.
    """
    model_type = opt['model_type']

    # dynamic instantiation
    for module in _model_modules:
        model_cls = getattr(module, model_type, None)
        if model_cls is not None:
            break
    if model_cls is None:
        raise ValueError(f'Model {model_type} is not found.')

    model = model_cls(opt)

    logger = get_root_logger()
    logger.info(f'Model [{model.__class__.__name__}] is created.')
    return model
