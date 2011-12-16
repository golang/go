// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "zasm_GOOS_GOARCH.h"

TEXT	_rt0_amd64_windows(SB),7,$-8
	MOVQ	$_rt0_amd64(SB), AX
	MOVQ	SP, DI
	JMP	AX

DATA  runtime·iswindows(SB)/4, $1
GLOBL runtime·iswindows(SB), $4
