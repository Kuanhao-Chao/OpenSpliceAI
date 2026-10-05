"""Copy the original model files from the separately installed SpliceAI package."""
from pathlib import Path
import shutil
import spliceai

source = Path(spliceai.__file__).parent/'models'
destination = Path('models/spliceai/SpliceAI_models_release')
destination.mkdir(parents=True, exist_ok=True)
for index in range(1,6):
    shutil.copy2(source/f'spliceai{index}.h5', destination/f'spliceai{index}.h5')
