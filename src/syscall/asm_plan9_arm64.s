// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "funcdata.h"

#define SYS_ERRSTR 41	/* from zsysnum_plan9.go */
#define SYS_SEEK 39	/* from zsysnum_plan9.go */

// System call support for plan9 on arm64

//func Syscall(trap, a1, a2, a3 uintptr) (r1, r2 uintptr, err ErrorString)
TEXT	·Syscall(SB),NOSPLIT,$168-64
	NO_LOCAL_POINTERS
	BL	runtime·entersyscall(SB)

	MOVD	trap+0(FP), R0
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4

	// move to syscall args
	MOVD	R2, sysargs-192(FP)
	MOVD	R3, sysargs-184(FP)
	MOVD	R4, sysargs-176(FP)

	SVC	$0

	// put return values into r1, r2, err
	MOVD	R0, r1+32(FP)
	MOVD	R1, r2+40(FP)
	MOVD	ZR, err+48(FP)

	// put error if needed
	CMP	$-1, R0
	BEQ	syscallerr
	BL	runtime·exitsyscall(SB)
	MOVD	$·emptystring+0(SB), R2
	B	syscallok
syscallerr:
	MOVD	$errbuf-128(SP), R2
	MOVD	$128, R3

	MOVD	$SYS_ERRSTR, R0
	MOVD	R2, err-192(FP)
	MOVD	R3, nerr-184(FP)
	SVC	$0

	BL	runtime·exitsyscall(SB)
	BL	runtime·gostring(SB)
	MOVD	$str-160(SP), R2
syscallok:
	MOVD	$err+48(FP), R1
	MOVD	0(R2), R3
	MOVD	8(R2), R4
	MOVD	R3, 0(R1)
	MOVD	R4, 8(R1)
	RET


//func Syscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2 uintptr, err ErrorString)
// Actually Syscall5 but the rest of the code expects it to be named Syscall6.
TEXT	·Syscall6(SB),NOSPLIT,$168-88
	NO_LOCAL_POINTERS
	BL	runtime·entersyscall(SB)

	MOVD	trap+0(FP), R1
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7

	// dereference pointers
	MOVD	R1, R0
	MOVD	R2, sysargs-192(FP)
	MOVD	R3, sysargs-184(FP)
	MOVD	R4, sysargs-176(FP)
	MOVD	R5, sysargs-168(FP)
	MOVD	R6, sysargs-160(FP)
	MOVD	R7, sysargs-152(FP)

	SVC	$0

	// put return value into r1, r2, err
	MOVD	R0, r1+56(FP)
	MOVD	R1, r2+64(FP)
	MOVD	ZR, err+72(FP)

	// put error if needed
	CMP	$-1, R0
	BEQ	syscall6err
	BL	runtime·exitsyscall(SB)
	MOVD	$·emptystring+0(SB), R2
	B	syscall6ok
syscall6err:
	MOVD	$errbuf-128(SP), R2
	MOVD	$128, R3

	MOVD	$SYS_ERRSTR, R0
	MOVD	R2, err-192(FP)
	MOVD	R3, nerr-184(FP)
	SVC	$0

	BL	runtime·exitsyscall(SB)
	BL	runtime·gostring(SB)
	MOVD	$str-160(SP), R2
syscall6ok:
	MOVD	$err+72(FP), R1
	MOVD	0(R2), R3
	MOVD	8(R2), R4
	MOVD	R3, 0(R1)
	MOVD	R4, 8(R1)
	RET

//func RawSyscall(trap, a1, a2, a3 uintptr) (r1, r2, err uintptr)
TEXT ·RawSyscall(SB),NOSPLIT,$24-56
	MOVD	trap+0(FP), R1
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4

	// move to syscall args
	MOVD	R1, R0
	MOVD	R2, sysargs-48(FP)
	MOVD	R3, sysargs-40(FP)
	MOVD	R4, sysargs-32(FP)

	SVC	$0

	// put return values into r1, r2, err
	MOVD	R0, r1+32(FP)
	MOVD	R0, r2+40(FP)
	MOVD	R0, err+48(FP)

	RET

//func RawSyscall6(trap, a1, a2, a3, a4, a5, a6 uintptr) (r1, r2, err uintptr)
// Actually RawSyscall5 but the rest of the code expects it to be named RawSyscall6.
TEXT	·RawSyscall6(SB),NOSPLIT,$48-80
	MOVD	trap+0(FP), R1
	MOVD	a1+8(FP), R2
	MOVD	a2+16(FP), R3
	MOVD	a3+24(FP), R4
	MOVD	a4+32(FP), R5
	MOVD	a5+40(FP), R6
	MOVD	a6+48(FP), R7

	// move to syscall args
	MOVD	R1, R0
	MOVD	R2, sysargs-64(FP)
	MOVD	R3, sysargs-56(FP)
	MOVD	R4, sysargs-48(FP)
	MOVD	R5, sysargs-40(FP)
	MOVD	R6, sysargs-32(FP)
	MOVD	R7, sysargs-24(FP)

	SVC	$0

	// put return values into r1, r2, err
	MOVD	R0, r1+56(FP)
	MOVD	R1, r2+64(FP)
	MOVD	ZR, err+72(FP)

	RET

//func seek(placeholder uintptr, fd int, offset int64, whence int) (newoffset int64, err string)
TEXT ·seek(SB),NOSPLIT,$168-56
	NO_LOCAL_POINTERS

	MOVD	$newoffset+32(FP), R0
	MOVWU	fd+8(FP), R2
	MOVD	offset+16(FP), R3
	MOVWU	whence+24(FP), R4

	// move to syscall args
	MOVD	R0, sysargs-192(FP)
	MOVWU	R2, sysargs-184(FP)
	MOVD	R3, sysargs-176(FP)
	MOVWU	R4, sysargs-168(FP)

	MOVD	$SYS_SEEK, R0
	SVC	$0

	// put err
	MOVD	ZR, err+40(FP)

	// put error if needed
	CMP	$-1, R0
	BEQ	syscallerr
	MOVD	$·emptystring+0(SB), R2
	B	syscallok
syscallerr:
	MOVD	R0, newoffset+32(FP)

	MOVD	$errbuf-128(SP), R2
	MOVD	$128, R3

	MOVD	$SYS_ERRSTR, R0
	MOVD	R2, err-192(FP)
	MOVD	R3, nerr-184(FP)
	SVC	$0

	BL	runtime·gostring(SB)
	MOVD	$str-160(SP), R2
syscallok:
	MOVD	$err+40(FP), R1
	MOVD	0(R2), R3
	MOVD	8(R2), R4
	MOVD	R3, 0(R1)
	MOVD	R4, 8(R1)
	RET


