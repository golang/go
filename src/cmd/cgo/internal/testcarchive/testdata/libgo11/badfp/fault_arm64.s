// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func Fault()
TEXT ·Fault(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$0x0034000000002000, R29
	MOVD	$0, R0
	MOVD	(R0), R0
	RET
