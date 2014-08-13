// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·Sin(SB),NOSPLIT,$0
	B	·sin(SB)

TEXT ·Cos(SB),NOSPLIT,$0
	B	·cos(SB)
