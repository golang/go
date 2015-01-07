// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !386,!amd64

#include "textflag.h"

TEXT ·cpuid(SB),NOSPLIT,$0-0
	RET
