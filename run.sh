#!/usr/bin/env bash
set -eou pipefail -o posix

# python -m src.exp.exp010.train --train-folds 0 1 2
# python -m src.exp.exp009.train --train-folds 0 1 2
# python -m src.exp.exp008.train --train-folds 0 1 2
# python -m src.exp.exp007.train --train-folds 0 1 2

# python -m src.exp.exp015.train --train-folds 0 1
# python -m src.exp.exp016.train --train-folds 0 1
# python -m src.exp.exp017.train --train-folds 0 1

python -m src.exp.exp020.train --train-folds 1 2
python -m src.exp.exp021.train --train-folds 1 2
python -m src.exp.exp022.train --train-folds 1 2
python -m src.exp.exp023.train --train-folds 1 2
