// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

TEXT ·Abs(SB),NOSPLIT,$0-16
	MOVD	x+0(FP), R3
	MOVD 	$((1<<63)-1), R4
	AND	R4, R3
	MOVD	R3, ret+8(FP)
	RETURN
