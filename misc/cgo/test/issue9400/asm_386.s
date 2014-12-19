#include "textflag.h"

TEXT ·RewindAndSetgid(SB),NOSPLIT,$0-0
	// Rewind stack pointer so anything that happens on the stack
	// will clobber the test pattern created by the caller
	ADDL	$(1024 * 8), SP

	// Ask signaller to setgid
	MOVL	$1, ·Baton(SB)

	// Wait for setgid completion
loop:
	PAUSE
	MOVL	·Baton(SB), AX
	CMPL	AX, $0
	JNE	loop

	// Restore stack
	SUBL	$(1024 * 8), SP
	RET
