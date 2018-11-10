// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 amd64 amd64p32 arm

#include "textflag.h"

TEXT ·Sinh(SB),NOSPLIT,$0
	JMP ·sinh(SB)

TEXT ·Cosh(SB),NOSPLIT,$0
	JMP ·cosh(SB)

TEXT ·Tanh(SB),NOSPLIT,$0
	JMP ·tanh(SB)

