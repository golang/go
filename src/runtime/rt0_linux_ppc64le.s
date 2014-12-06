#include "textflag.h"

TEXT _rt0_ppc64le_linux(SB),NOSPLIT,$0
	BR _main<>(SB)

TEXT _main<>(SB),NOSPLIT,$-8
	MOVD 0(R1), R3 // argc
	ADD $8, R1, R4 // argv
	BR main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R31
	MOVD	R31, CTR
	BR	(CTR)
