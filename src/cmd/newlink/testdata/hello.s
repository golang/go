TEXT _rt0_go(SB),7,$0
	MOVL $1, DI
	MOVL $hello<>(SB), SI
	MOVL $12, DX
	MOVL $0x2000004, AX
	SYSCALL
	MOVL $0, DI
	MOVL $0x2000001, AX
	SYSCALL
	RET

DATA hello<>+0(SB)/4, $"hell"
DATA hello<>+4(SB)/4, $"o wo"
DATA hello<>+8(SB)/4, $"rld\n"
GLOBL hello<>(SB), $12
