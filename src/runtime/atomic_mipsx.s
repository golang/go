// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

TEXT ·publicationBarrier(SB),NOSPLIT,$0
	SYNC
	RET
