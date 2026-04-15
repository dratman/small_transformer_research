#!/usr/bin/env python3
"""
Strip ANSI escape codes and terminal control sequences from raw Claude Code
session logs, producing readable plain text.

Usage:
    python py/clean_session_log.py claude_code_sessions/SESSION_*.raw.txt
    python py/clean_session_log.py claude_code_sessions/  # process all .raw.txt in dir

Output files are written alongside the inputs with .raw replaced by .clean:
    SESSION_2026_04_06_2227.raw.txt -> SESSION_2026_04_06_2227.clean.txt
"""

import re
import sys
import os
import glob


def clean_session_text(text):
    # Remove ANSI escape sequences: CSI (ESC[...), OSC (ESC]...), and others
    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text)   # CSI sequences
    text = re.sub(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)', '', text)  # OSC sequences
    text = re.sub(r'\x1b[()][0-9A-Za-z]', '', text)      # charset selection
    text = re.sub(r'\x1b[>=<]', '', text)                 # keypad/cursor mode
    text = re.sub(r'\x1b\[[\?]?[0-9;]*[a-zA-Z]', '', text)  # private CSI
    text = re.sub(r'\x1b.', '', text)                     # any remaining ESC + char

    # Remove other control characters except newline and tab
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Collapse runs of blank lines to at most two
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Remove lines that are only whitespace
    lines = text.split('\n')
    lines = [line.rstrip() for line in lines]
    text = '\n'.join(lines)

    return text


def process_file(input_path):
    output_path = input_path.replace('.raw.', '.clean.')
    if output_path == input_path:
        output_path = input_path.rsplit('.', 1)[0] + '.clean.txt'

    with open(input_path, 'r', errors='replace') as f:
        raw = f.read()

    cleaned = clean_session_text(raw)

    with open(output_path, 'w') as f:
        f.write(cleaned)

    raw_size = os.path.getsize(input_path)
    clean_size = os.path.getsize(output_path)
    print(f"{os.path.basename(input_path)}: {raw_size:,} -> {clean_size:,} bytes ({100*clean_size//raw_size}%) -> {os.path.basename(output_path)}")


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            paths.extend(sorted(glob.glob(os.path.join(arg, '*.raw.txt'))))
        else:
            paths.append(arg)

    if not paths:
        print("No .raw.txt files found.")
        sys.exit(1)

    for path in paths:
        process_file(path)


if __name__ == '__main__':
    main()
