// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Dim(SB),NOSPLIT,$0
	JMP ·dim(SB)

TEXT ·Max(SB),NOSPLIT,$0
	JMP ·max(SB)

TEXT ·Min(SB),NOSPLIT,$0
	JMP ·min(SB)
