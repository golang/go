// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x && gc

#include "textflag.h"

#define PSALAA            1208(R0)
#define GTAB64(x)           80(x)
#define LCA64(x)            88(x)
#define CAA(x)               8(x)
#define EDCHPXV(x)        1016(x)       // in the CAA
#define SAVSTACK_ASYNC(x)  336(x)       // in the LCA

// SS_*, where x=SAVSTACK_ASYNC
#define SS_LE(x)             0(x)
#define SS_GO(x)             8(x)
#define SS_ERRNO(x)         16(x)
#define SS_ERRNOJR(x)       20(x)

#define LE_CALL BYTE $0x0D; BYTE $0x76; // BL R7, R6

TEXT ·clearErrno(SB),NOSPLIT,$0-0
	BL	addrerrno<>(SB)
	MOVD	$0, 0(R3)
	RET

// Returns the address of errno in R3.
TEXT addrerrno<>(SB),NOSPLIT|NOFRAME,$0-0
	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get __errno FuncDesc.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	ADD	$(0x156*16), R9
	LMG	0(R9), R5, R6

	// Switch to saved LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Call __errno function.
	LE_CALL
	NOPH

	// Switch back to Go stack.
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.
	RET

TEXT ·syscall_syscall(SB),NOSPLIT,$0-56
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+32(FP)
	MOVD	R0, r2+40(FP)
	MOVD	R0, err+48(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	addrerrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+48(FP)
done:
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·syscall_rawsyscall(SB),NOSPLIT,$0-56
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+32(FP)
	MOVD	R0, r2+40(FP)
	MOVD	R0, err+48(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	addrerrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+48(FP)
done:
	RET

TEXT ·syscall_syscall6(SB),NOSPLIT,$0-80
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Fill in parameter list.
	MOVD	a4+32(FP), R12
	MOVD	R12, (2176+24)(R4)
	MOVD	a5+40(FP), R12
	MOVD	R12, (2176+32)(R4)
	MOVD	a6+48(FP), R12
	MOVD	R12, (2176+40)(R4)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+56(FP)
	MOVD	R0, r2+64(FP)
	MOVD	R0, err+72(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	addrerrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+72(FP)
done:
	BL	runtime·exitsyscall(SB)
	RET

TEXT ·syscall_rawsyscall6(SB),NOSPLIT,$0-80
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Fill in parameter list.
	MOVD	a4+32(FP), R12
	MOVD	R12, (2176+24)(R4)
	MOVD	a5+40(FP), R12
	MOVD	R12, (2176+32)(R4)
	MOVD	a6+48(FP), R12
	MOVD	R12, (2176+40)(R4)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+56(FP)
	MOVD	R0, r2+64(FP)
	MOVD	R0, err+72(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	·rrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+72(FP)
done:
	RET

TEXT ·syscall_syscall9(SB),NOSPLIT,$0
	BL	runtime·entersyscall(SB)
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Fill in parameter list.
	MOVD	a4+32(FP), R12
	MOVD	R12, (2176+24)(R4)
	MOVD	a5+40(FP), R12
	MOVD	R12, (2176+32)(R4)
	MOVD	a6+48(FP), R12
	MOVD	R12, (2176+40)(R4)
	MOVD	a7+56(FP), R12
	MOVD	R12, (2176+48)(R4)
	MOVD	a8+64(FP), R12
	MOVD	R12, (2176+56)(R4)
	MOVD	a9+72(FP), R12
	MOVD	R12, (2176+64)(R4)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+80(FP)
	MOVD	R0, r2+88(FP)
	MOVD	R0, err+96(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	addrerrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+96(FP)
done:
        BL	runtime·exitsyscall(SB)
        RET

TEXT ·syscall_rawsyscall9(SB),NOSPLIT,$0
	MOVD	a1+8(FP), R1
	MOVD	a2+16(FP), R2
	MOVD	a3+24(FP), R3

	// Get library control area (LCA).
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8

	// Get function.
	MOVD	CAA(R8), R9
	MOVD	EDCHPXV(R9), R9
	MOVD	trap+0(FP), R5
	SLD	$4, R5
	ADD	R5, R9
	LMG	0(R9), R5, R6

	// Restore LE stack.
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R4
	MOVD	$0, 0(R9)

	// Fill in parameter list.
	MOVD	a4+32(FP), R12
	MOVD	R12, (2176+24)(R4)
	MOVD	a5+40(FP), R12
	MOVD	R12, (2176+32)(R4)
	MOVD	a6+48(FP), R12
	MOVD	R12, (2176+40)(R4)
	MOVD	a7+56(FP), R12
	MOVD	R12, (2176+48)(R4)
	MOVD	a8+64(FP), R12
	MOVD	R12, (2176+56)(R4)
	MOVD	a9+72(FP), R12
	MOVD	R12, (2176+64)(R4)

	// Call function.
	LE_CALL
	NOPH
	XOR	R0, R0      // Restore R0 to $0.
	MOVD	R4, 0(R9)   // Save stack pointer.

	MOVD	R3, r1+80(FP)
	MOVD	R0, r2+88(FP)
	MOVD	R0, err+96(FP)
	MOVW	R3, R4
	CMP	R4, $-1
	BNE	done
	BL	addrerrno<>(SB)
	MOVWZ	0(R3), R3
	MOVD	R3, err+96(FP)
done:
	RET

// func svcCall(fnptr unsafe.Pointer, argv *unsafe.Pointer, dsa *uint64)
TEXT ·svcCall(SB),NOSPLIT,$0
	BL	runtime·save_g(SB)   // Save g and stack pointer
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	R15, 0(R9)

	MOVD	argv+8(FP), R1       // Move function arguments into registers
	MOVD	dsa+16(FP), g
	MOVD	fnptr+0(FP), R15

	BYTE	$0x0D                // Branch to function
	BYTE	$0xEF

	BL	runtime·load_g(SB)   // Restore g and stack pointer
	MOVW	PSALAA, R8
	MOVD	LCA64(R8), R8
	MOVD	SAVSTACK_ASYNC(R8), R9
	MOVD	0(R9), R15

	RET

// func svcLoad(name *byte) unsafe.Pointer
TEXT ·svcLoad(SB),NOSPLIT,$0
	MOVD	R15, R2          // Save go stack pointer
	MOVD	name+0(FP), R0   // Move SVC args into registers
	MOVD	$0x80000000, R1
	MOVD	$0, R15
	BYTE	$0x0A            // SVC 08 LOAD
	BYTE	$0x08
	MOVW	R15, R3          // Save return code from SVC
	MOVD	R2, R15          // Restore go stack pointer
	CMP	R3, $0           // Check SVC return code
	BNE	error

	MOVD	$-2, R3          // Reset last bit of entry point to zero
	AND	R0, R3
	MOVD	R3, addr+8(FP)   // Return entry point returned by SVC
	CMP	R0, R3           // Check if last bit of entry point was set
	BNE	done

	MOVD	R15, R2          // Save go stack pointer
	MOVD	$0, R15          // Move SVC args into registers (entry point still in r0 from SVC 08)
	BYTE	$0x0A            // SVC 09 DELETE
	BYTE	$0x09
	MOVD	R2, R15          // Restore go stack pointer

error:
	MOVD	$0, addr+8(FP)   // Return 0 on failure
done:
	XOR	R0, R0           // Reset r0 to 0
	RET

// func svcUnload(name *byte, fnptr unsafe.Pointer) int64
TEXT ·svcUnload(SB),NOSPLIT,$0
	MOVD	R15, R2          // Save go stack pointer
	MOVD	name+0(FP), R0   // Move SVC args into registers
	MOVD	addr+8(FP), R15
	BYTE	$0x0A            // SVC 09
	BYTE	$0x09
	XOR	R0, R0           // Reset r0 to 0
	MOVD	R15, R1          // Save SVC return code
	MOVD	R2, R15          // Restore go stack pointer
	MOVD	R1, rc+0(FP)     // Return SVC return code
	RET

// func gettid() uint64
TEXT ·gettid(SB), NOSPLIT, $0
	// Get library control area (LCA).
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8

	// Get CEECAATHDID
	MOVD CAA(R8), R9
	MOVD 0x3D0(R9), R9
	MOVD R9, ret+0(FP)

	RET
