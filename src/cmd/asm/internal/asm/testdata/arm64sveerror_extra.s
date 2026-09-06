// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input extends auto-generated arm64sveerror.s test suite
// with manually added tests.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$-8
	// Only SP-relative byte offsets are materialized; GP-based byte
	// offsets and the zero-offset (Rn) form remain unsupported.
	ZSTR Z3, 8(R2)                     // ERROR "illegal combination from SVE"
	ZLDR (R7), Z9                      // ERROR "illegal combination from SVE"
	RET
