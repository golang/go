// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// s390x can copy/zero 1-256 bytes with a single instruction,
// so there's no need for these, except to satisfy the prototypes
// in stubs.go.

TEXT runtime·duffzero(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$0, 2(R0)
	RET

TEXT runtime·duffcopy(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$0, 2(R0)
	RET
