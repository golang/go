// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

//
// System call support for ARM, FreeBSD
//

// func Syscall(trap, a1, a2, a3 uintptr) (r1, r2, errno uintptr);
// func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, errno uintptr);
// func Syscall9(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2, errno uintptr)

TEXT	·Syscall(SB),NOSPLIT,$0-28
	BL runtime·entersyscall(SB)
	MOVW 0(FP), R7 // syscall number
	MOVW 4(FP), R0 // a1
	MOVW 8(FP), R1 // a2
	MOVW 12(FP), R2 // a3
	SWI $0 // syscall
	MOVW $0, R2
	BCS error
	MOVW R0, 16(FP) // r1
	MOVW R1, 20(FP) // r2
	MOVW R2, 24(FP) // errno
	BL runtime·exitsyscall(SB)
	RET
error:
	MOVW $-1, R3
	MOVW R3, 16(FP) // r1
	MOVW R2, 20(FP) // r2
	MOVW R0, 24(FP) // errno
	BL runtime·exitsyscall(SB)
	RET

TEXT	·Syscall6(SB),NOSPLIT,$0-40
	BL runtime·entersyscall(SB)
	MOVW 0(FP), R7 // syscall number
	MOVW 4(FP), R0 // a1
	MOVW 8(FP), R1 // a2
	MOVW 12(FP), R2 // a3
	MOVW 16(FP), R3 // a4
	MOVW R13, R4
	MOVW $20(FP), R13 // a5 to a6 are passed on stack
	SWI $0 // syscall
	MOVW R4, R13
	MOVW $0, R2
	BCS error6
	MOVW R0, 28(FP) // r1
	MOVW R1, 32(FP) // r2
	MOVW R2, 36(FP) // errno
	BL runtime·exitsyscall(SB)
	RET
error6:
	MOVW $-1, R3
	MOVW R3, 28(FP) // r1
	MOVW R2, 32(FP) // r2
	MOVW R0, 36(FP) // errno
	BL runtime·exitsyscall(SB)
	RET

TEXT	·Syscall9(SB),NOSPLIT,$0-52
	BL runtime·entersyscall(SB)
	MOVW 0(FP), R7 // syscall number
	MOVW 4(FP), R0 // a1
	MOVW 8(FP), R1 // a2
	MOVW 12(FP), R2 // a3
	MOVW 16(FP), R3 // a4
	MOVW R13, R4
	MOVW $20(FP), R13 // a5 to a9 are passed on stack
	SWI $0 // syscall
	MOVW R4, R13
	MOVW $0, R2
	BCS error9
	MOVW R0, 40(FP) // r1
	MOVW R1, 44(FP) // r2
	MOVW R2, 48(FP) // errno
	BL runtime·exitsyscall(SB)
	RET
error9:
	MOVW $-1, R3
	MOVW R3, 40(FP) // r1
	MOVW R2, 44(FP) // r2
	MOVW R0, 48(FP) // errno
	BL runtime·exitsyscall(SB)
	RET

TEXT	·RawSyscall(SB),NOSPLIT,$0-28
	MOVW 0(FP), R7 // syscall number
	MOVW 4(FP), R0 // a1
	MOVW 8(FP), R1 // a2
	MOVW 12(FP), R2 // a3
	SWI $0 // syscall
	MOVW $0, R2
	BCS errorr
	MOVW R0, 16(FP) // r1
	MOVW R1, 20(FP) // r2
	MOVW R2, 24(FP) // errno
	RET
errorr:
	MOVW $-1, R3
	MOVW R3, 16(FP) // r1
	MOVW R2, 20(FP) // r2
	MOVW R0, 24(FP) // errno
	RET

TEXT	·RawSyscall6(SB),NOSPLIT,$0-40
	MOVW 0(FP), R7 // syscall number
	MOVW 4(FP), R0 // a1
	MOVW 8(FP), R1 // a2
	MOVW 12(FP), R2 // a3
	MOVW 16(FP), R3 // a4
	MOVW R13, R4
	MOVW $20(FP), R13 // a5 to a6 are passed on stack
	SWI $0 // syscall
	MOVW R4, R13
	MOVW $0, R2
	BCS errorr6
	MOVW R0, 28(FP) // r1
	MOVW R1, 32(FP) // r2
	MOVW R2, 36(FP) // errno
	RET
errorr6:
	MOVW $-1, R3
	MOVW R3, 28(FP) // r1
	MOVW R2, 32(FP) // r2
	MOVW R0, 36(FP) // errno
	RET
