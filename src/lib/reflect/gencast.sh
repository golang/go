#!/bin/bash
# Copyright 2009 The Go Authors.  All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

awk '
BEGIN {
	print("// Conversion operators - really just casts")
	print("// *** Created by gencast.sh - Do Not Edit ***\n")}
{
	print("TEXT reflect·AddrToPtr" $0 "(SB),7,$-8")
	print("\tMOVQ	8(SP), AX")
	print("\tMOVQ	AX, 16(SP)")
	print("\tRET")
	print("")
	print("TEXT reflect·Ptr" $0 "ToAddr(SB),7,$-8")
	print("\tMOVQ	8(SP), AX")
	print("\tMOVQ	AX, 16(SP)")
	print("\tRET")
	print("")
}
' > cast_$GOARCH.s << '!'
Addr
Int
Int8
Int16
Int32
Int64
Uint
Uint8
Uint16
Uint32
Uint64
Float
Float32
Float64
Float80
String
!
