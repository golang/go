// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func syncIcache(p uintptr)
TEXT main·syncIcache(SB), NOSPLIT|NOFRAME, $0-0
	SYNC
	MOVD (R3), R3
	ICBI (R3)
	ISYNC
	RET
