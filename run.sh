#!/bin/bash

VENV_DIR="venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --upgrade pip -q > /dev/null 2>&1
pip install -r requirements.txt -q > /dev/null 2>&1

# 1. Launch the ML pipeline
python main.py "$@"

# Push the terminal cursor down to prevent the table from overlapping the progress bars
echo -e "\n\n\n\n\n\n\n\n\n\n"

# 2. Verify Unitary Matrices (Silently passes, only prints on FAILURE)
if ! pytest tests/test_unitary_outputs.py -q > /dev/null 2>&1; then
    echo -e "\n❌ Unitary verification FAILED. Displaying error report:\n"
    pytest tests/test_unitary_outputs.py -v
fi

# 3. Aggregate Multirun Data and Print Table
if [ -d "multirun" ]; then
    LATEST_RUN=$(ls -td multirun/*/* 2>/dev/null | head -1)
    if [ ! -z "$LATEST_RUN" ]; then
        python aggregate.py "$LATEST_RUN"
    fi
fi

deactivate
