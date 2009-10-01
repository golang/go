// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls for arm, Linux
//

#define SYS_BASE 0x00900000
#define SYS_syscall (SYS_BASE+113);

// TODO(kaib): handle error returns

// func Syscall(syscall uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);

TEXT	syscall·Syscall(SB),7,$0
	BL		sys·entersyscall(SB)
	MOVW	4(SP), R7
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	SWI		$SYS_syscall
	MOVW	R0, 20(SP)	// r1
	MOVW	R1, 24(SP)	// r2
	MOVW	$0, 28(SP)	// errno
	BL		sys·exitsyscall(SB)
	RET

// func Syscall6(trap uintptr, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr);
// Actually Syscall5 but the rest of the code expects it to be named Syscall6.
TEXT	syscall·Syscall6(SB),7,$0
	BL		sys·entersyscall(SB)
	MOVW	4(SP), R7	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	MOVW	20(SP), R3
	MOVW	24(SP), R4
	MOVW	28(SP), R5
	SWI		$SYS_syscall
	MOVW	R0, 32(SP)	// r1
	MOVW	R1, 36(SP)	// r2
	MOVW	$0, 40(SP)	// errno
	BL		sys·exitsyscall(SB)
	RET

// func RawSyscall(trap uintptr, a1, a2, a3 uintptr) (r1, r2, err uintptr);
TEXT syscall·RawSyscall(SB),7,$0
	MOVW	4(SP), R7	// syscall entry
	MOVW	8(SP), R0
	MOVW	12(SP), R1
	MOVW	16(SP), R2
	SWI		$SYS_syscall
	MOVW	R0, 20(SP)	// r1
	MOVW	R1, 24(SP)	// r2
	MOVW	$0, 28(SP)	// errno
	RET
