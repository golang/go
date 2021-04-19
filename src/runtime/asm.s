// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// funcdata for functions with no local variables in frame.
// Define two zero-length bitmaps, because the same index is used
// for the local variables as for the argument frame, and assembly
// frames have two argument bitmaps, one without results and one with results.
DATA runtime·no_pointers_stackmap+0x00(SB)/4, $2
DATA runtime·no_pointers_stackmap+0x04(SB)/4, $0
GLOBL runtime·no_pointers_stackmap(SB),RODATA, $8

GLOBL runtime·mheap_(SB), NOPTR, $0
GLOBL runtime·memstats(SB), NOPTR, $0
