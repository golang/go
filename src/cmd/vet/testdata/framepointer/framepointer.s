// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64
// +build amd64

#include "textflag.h"

TEXT ·f(SB),0,$0-0
	MOVQ $0, BP // ERROR "frame pointer is clobbered before saving"
	RET
