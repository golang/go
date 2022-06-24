// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "textflag.h"

TEXT ·publicationBarrier(SB),NOSPLIT|NOFRAME,$0-0
	// LWSYNC is the "export" barrier recommended by Power ISA
	// v2.07 book II, appendix B.2.2.2.
	// LWSYNC is a load/load, load/store, and store/store barrier.
	LWSYNC
	RET
