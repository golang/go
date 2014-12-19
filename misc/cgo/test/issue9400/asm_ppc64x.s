// +build ppc64 ppc64le

#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$-8-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADD	$(1024 * 8), R1

	// Ask signaller to setgid
	MOVW	$1, R3
	SYNC
	MOVW	R3, ·Baton(SB)

	// Wait for setgid completion
loop:
	SYNC
	MOVW	·Baton(SB), R3
	CMP	R3, $0
	// Hint that we're in a spin loop
	OR	R1, R1, R1
	BNE	loop
	ISYNC

	// Restore stack
	SUB	$(1024 * 8), R1
	RET
