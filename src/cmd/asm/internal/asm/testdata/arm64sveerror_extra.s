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
	// A right shift is at most the element size, a narrowing one at most
	// the narrow element size, and a left shift less than the element size.
	ZASR $9, Z7.B, Z23.B               // ERROR "illegal combination from SVE"
	ZLSR $65, Z23.D, P1.M, Z23.D       // ERROR "illegal combination from SVE"
	ZSHRNB $17, Z7.S, Z23.H            // ERROR "illegal combination from SVE"
	ZLSL $8, Z7.B, Z23.B               // ERROR "illegal combination from SVE"
	ZLSL $64, Z23.D, P1.M, Z23.D       // ERROR "illegal combination from SVE"
	RET
