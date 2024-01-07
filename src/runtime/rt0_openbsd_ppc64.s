// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_ppc64_openbsd(SB),NOSPLIT,$0
	BR	main(SB)

TEXT main(SB),NOSPLIT,$-8
	// Make sure R0 is zero before _main
	XOR	R0, R0

	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)
