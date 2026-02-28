// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#define NOSPLIT 7

TEXT ·asmMain(SB),0,$0-0
	CALL ·startSelf(SB)
	CALL ·startChain(SB)
	CALL ·startRec(SB)
	RET

// Test reporting of basic over-the-limit
TEXT ·startSelf(SB),NOSPLIT,$1000-0
	RET

// Test reporting of multiple over-the-limit chains
TEXT ·startChain(SB),NOSPLIT,$16-0
	CALL ·chain0(SB)
	CALL ·chain1(SB)
	CALL ·chain2(SB)
	RET
TEXT ·chain0(SB),NOSPLIT,$32-0
	CALL ·chainEnd(SB)
	RET
TEXT ·chain1(SB),NOSPLIT,$48-0 // Doesn't go over
	RET
TEXT ·chain2(SB),NOSPLIT,$64-0
	CALL ·chainEnd(SB)
	RET
TEXT ·chainEnd(SB),NOSPLIT,$1000-0 // Should be reported twice
	RET

// Test reporting of rootless recursion
TEXT ·startRec(SB),NOSPLIT,$0-0
	CALL ·startRec0(SB)
	RET
TEXT ·startRec0(SB),NOSPLIT,$0-0
	CALL ·startRec(SB)
	RET
