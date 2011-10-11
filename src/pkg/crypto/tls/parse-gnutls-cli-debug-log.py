# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

# This code is used to parse the debug log from gnutls-cli and generate a
# script of the handshake. This script is included in handshake_server_test.go.
# See the comments there for details.

import sys

blocks = []

READ = 1
WRITE = 2

currentBlockType = 0
currentBlock = []
for line in sys.stdin.readlines():
        line = line[:-1]
        if line.startswith("|<7>| WRITE: "):
                if currentBlockType != WRITE:
                        if len(currentBlock) > 0:
                                blocks.append(currentBlock)
                        currentBlock = []
                        currentBlockType = WRITE
        elif line.startswith("|<7>| READ: "):
                if currentBlockType != READ:
                        if len(currentBlock) > 0:
                                blocks.append(currentBlock)
                        currentBlock = []
                        currentBlockType = READ
        elif line.startswith("|<7>| 0"):
                line = line[13:]
                line = line.strip()
                bs = line.split()
                for b in bs:
                        currentBlock.append(int(b, 16))
	elif line.startswith("|<7>| RB-PEEK: Read 1 bytes"):
		currentBlock = currentBlock[:-1]

if len(currentBlock) > 0:
        blocks.append(currentBlock)

for block in blocks:
        sys.stdout.write("\t{\n")

        i = 0
        for b in block:
                if i % 8 == 0:
                        sys.stdout.write("\t\t")
                sys.stdout.write("0x%02x," % b)
                if i % 8 == 7:
                        sys.stdout.write("\n")
                else:
                        sys.stdout.write(" ")
                i += 1
        sys.stdout.write("\n\t},\n\n")
