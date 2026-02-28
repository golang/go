// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·libc_getentropy_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_getentropy(SB)
