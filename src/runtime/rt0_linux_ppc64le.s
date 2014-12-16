#include "textflag.h"

TEXT _rt0_ppc64le_linux(SB),NOSPLIT,$0
	BR _main<>(SB)

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	//
	// In a dynamically linked binary, r3 contains argc, r4
	// contains argv, r5 contains envp, r6 contains auxv, and r13
	// contains the TLS pointer.
	//
	// Figure out which case this is by looking at r4: if it's 0,
	// we're statically linked; otherwise we're dynamically
	// linked.
	CMP	R0, R4
	BNE	dlink

	// Statically linked
	MOVD	0(R1), R3 // argc
	ADD	$8, R1, R4 // argv
	MOVD	$runtime·tls0(SB), R13 // TLS
	ADD	$0x7000, R13

dlink:
	BR	main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R31
	MOVD	R31, CTR
	BR	(CTR)
