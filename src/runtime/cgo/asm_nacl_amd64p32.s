// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 */
TEXT crosscall2(SB),NOSPLIT,$0
	INT $3
	RET
