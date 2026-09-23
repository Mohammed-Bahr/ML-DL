#!/bin/bash

# Define the output file name
OUTPUT_FILE="history.txt"

# 1. Force the current session to write its in-memory history to the history file
history -a

# 2. Since history is disabled by default in non-interactive scripts,
# we point to the main history file and force-enable the history mechanism.
HISTFILE=~/.bash_history
set -o history

# 3. Read the history file and write all contents to the target file
history > "$OUTPUT_FILE"

echo "All bash history has been successfully saved to $OUTPUT_FILE"
