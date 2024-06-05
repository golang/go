// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x && gc

#include "textflag.h"

#define PSALAA            1208(R0)
#define GTAB64(x)           80(x)
#define LCA64(x)            88(x)
#define SAVSTACK_ASYNC(x)  336(x) // in the LCA
#define CAA(x)               8(x)
#define CEECAATHDID(x)     976(x) // in the CAA
#define EDCHPXV(x)        1016(x) // in the CAA
#define GOCB(x)           1104(x) // in the CAA

// SS_*, where x=SAVSTACK_ASYNC
#define SS_LE(x)             0(x)
#define SS_GO(x)             8(x)
#define SS_ERRNO(x)         16(x)
#define SS_ERRNOJR(x)       20(x)

// Function Descriptor Offsets
#define __errno  0x156*16
#define __err2ad 0x16C*16

// Call Instructions
#define LE_CALL    BYTE $0x0D; BYTE $0x76 // BL R7, R6
#define SVC_LOAD   BYTE $0x0A; BYTE $0x08 // SVC 08 LOAD
#define SVC_DELETE BYTE $0x0A; BYTE $0x09 // SVC 09 DELETE

DATA zosLibVec<>(SB)/8, $0
GLOBL zosLibVec<>(SB), NOPTR, $8

TEXT ·initZosLibVec(SB), NOSPLIT|NOFRAME, $0-0
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8
	MOVD CAA(R8), R8
	MOVD EDCHPXV(R8), R8
	MOVD R8, zosLibVec<>(SB)
	RET

TEXT ·GetZosLibVec(SB), NOSPLIT|NOFRAME, $0-0
	MOVD zosLibVec<>(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·clearErrno(SB), NOSPLIT, $0-0
	BL   addrerrno<>(SB)
	MOVD $0, 0(R3)
	RET

// Returns the address of errno in R3.
TEXT addrerrno<>(SB), NOSPLIT|NOFRAME, $0-0
	// Get library control area (LCA).
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8

	// Get __errno FuncDesc.
	MOVD CAA(R8), R9
	MOVD EDCHPXV(R9), R9
	ADD  $(__errno), R9
	LMG  0(R9), R5, R6

	// Switch to saved LE stack.
	MOVD SAVSTACK_ASYNC(R8), R9
	MOVD 0(R9), R4
	MOVD $0, 0(R9)

	// Call __errno function.
	LE_CALL
	NOPH

	// Switch back to Go stack.
	XOR  R0, R0    // Restore R0 to $0.
	MOVD R4, 0(R9) // Save stack pointer.
	RET

// func svcCall(fnptr unsafe.Pointer, argv *unsafe.Pointer, dsa *uint64)
TEXT ·svcCall(SB), NOSPLIT, $0
	BL   runtime·save_g(SB)     // Save g and stack pointer
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8
	MOVD SAVSTACK_ASYNC(R8), R9
	MOVD R15, 0(R9)

	MOVD argv+8(FP), R1   // Move function arguments into registers
	MOVD dsa+16(FP), g
	MOVD fnptr+0(FP), R15

	BYTE $0x0D // Branch to function
	BYTE $0xEF

	BL   runtime·load_g(SB)     // Restore g and stack pointer
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8
	MOVD SAVSTACK_ASYNC(R8), R9
	MOVD 0(R9), R15

	RET

// func svcLoad(name *byte) unsafe.Pointer
TEXT ·svcLoad(SB), NOSPLIT, $0
	MOVD R15, R2         // Save go stack pointer
	MOVD name+0(FP), R0  // Move SVC args into registers
	MOVD $0x80000000, R1
	MOVD $0, R15
	SVC_LOAD
	MOVW R15, R3         // Save return code from SVC
	MOVD R2, R15         // Restore go stack pointer
	CMP  R3, $0          // Check SVC return code
	BNE  error

	MOVD $-2, R3       // Reset last bit of entry point to zero
	AND  R0, R3
	MOVD R3, ret+8(FP) // Return entry point returned by SVC
	CMP  R0, R3        // Check if last bit of entry point was set
	BNE  done

	MOVD R15, R2 // Save go stack pointer
	MOVD $0, R15 // Move SVC args into registers (entry point still in r0 from SVC 08)
	SVC_DELETE
	MOVD R2, R15 // Restore go stack pointer

error:
	MOVD $0, ret+8(FP) // Return 0 on failure

done:
	XOR R0, R0 // Reset r0 to 0
	RET

// func svcUnload(name *byte, fnptr unsafe.Pointer) int64
TEXT ·svcUnload(SB), NOSPLIT, $0
	MOVD R15, R2          // Save go stack pointer
	MOVD name+0(FP), R0   // Move SVC args into registers
	MOVD fnptr+8(FP), R15
	SVC_DELETE
	XOR  R0, R0           // Reset r0 to 0
	MOVD R15, R1          // Save SVC return code
	MOVD R2, R15          // Restore go stack pointer
	MOVD R1, ret+16(FP)   // Return SVC return code
	RET

// func gettid() uint64
TEXT ·gettid(SB), NOSPLIT, $0
	// Get library control area (LCA).
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8

	// Get CEECAATHDID
	MOVD CAA(R8), R9
	MOVD CEECAATHDID(R9), R9
	MOVD R9, ret+0(FP)

	RET

//
// Call LE function, if the return is -1
// errno and errno2 is retrieved
//
TEXT ·CallLeFuncWithErr(SB), NOSPLIT, $0
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8
	MOVD CAA(R8), R9
	MOVD g, GOCB(R9)

	// Restore LE stack.
	MOVD SAVSTACK_ASYNC(R8), R9 // R9-> LE stack frame saving address
	MOVD 0(R9), R4              // R4-> restore previously saved stack frame pointer

	MOVD parms_base+8(FP), R7 // R7 -> argument array
	MOVD parms_len+16(FP), R8 // R8 number of arguments

	//  arg 1 ---> R1
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	MOVD 0(R7), R1

	//  arg 2 ---> R2
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	ADD  $8, R7
	MOVD 0(R7), R2

	//  arg 3 --> R3
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	ADD  $8, R7
	MOVD 0(R7), R3

	CMP  R8, $0
	BEQ  docall
	MOVD $2176+16, R6 // starting LE stack address-8 to store 4th argument

repeat:
	ADD  $8, R7
	MOVD 0(R7), R0      // advance arg pointer by 8 byte
	ADD  $8, R6         // advance LE argument address by 8 byte
	MOVD R0, (R4)(R6*1) // copy argument from go-slice to le-frame
	SUB  $1, R8
	CMP  R8, $0
	BNE  repeat

docall:
	MOVD funcdesc+0(FP), R8 // R8-> function descriptor
	LMG  0(R8), R5, R6
	MOVD $0, 0(R9)          // R9 address of SAVSTACK_ASYNC
	LE_CALL                 // balr R7, R6 (return #1)
	NOPH
	MOVD R3, ret+32(FP)
	CMP  R3, $-1            // compare result to -1
	BNE  done

	// retrieve errno and errno2
	MOVD  zosLibVec<>(SB), R8
	ADD   $(__errno), R8
	LMG   0(R8), R5, R6
	LE_CALL                   // balr R7, R6 __errno (return #3)
	NOPH
	MOVWZ 0(R3), R3
	MOVD  R3, err+48(FP)
	MOVD  zosLibVec<>(SB), R8
	ADD   $(__err2ad), R8
	LMG   0(R8), R5, R6
	LE_CALL                   // balr R7, R6 __err2ad (return #2)
	NOPH
	MOVW  (R3), R2            // retrieve errno2
	MOVD  R2, errno2+40(FP)   // store in return area

done:
	MOVD R4, 0(R9)            // Save stack pointer.
	RET

//
// Call LE function, if the return is 0
// errno and errno2 is retrieved
//
TEXT ·CallLeFuncWithPtrReturn(SB), NOSPLIT, $0
	MOVW PSALAA, R8
	MOVD LCA64(R8), R8
	MOVD CAA(R8), R9
	MOVD g, GOCB(R9)

	// Restore LE stack.
	MOVD SAVSTACK_ASYNC(R8), R9 // R9-> LE stack frame saving address
	MOVD 0(R9), R4              // R4-> restore previously saved stack frame pointer

	MOVD parms_base+8(FP), R7 // R7 -> argument array
	MOVD parms_len+16(FP), R8 // R8 number of arguments

	//  arg 1 ---> R1
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	MOVD 0(R7), R1

	//  arg 2 ---> R2
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	ADD  $8, R7
	MOVD 0(R7), R2

	//  arg 3 --> R3
	CMP  R8, $0
	BEQ  docall
	SUB  $1, R8
	ADD  $8, R7
	MOVD 0(R7), R3

	CMP  R8, $0
	BEQ  docall
	MOVD $2176+16, R6 // starting LE stack address-8 to store 4th argument

repeat:
	ADD  $8, R7
	MOVD 0(R7), R0      // advance arg pointer by 8 byte
	ADD  $8, R6         // advance LE argument address by 8 byte
	MOVD R0, (R4)(R6*1) // copy argument from go-slice to le-frame
	SUB  $1, R8
	CMP  R8, $0
	BNE  repeat

docall:
	MOVD funcdesc+0(FP), R8 // R8-> function descriptor
	LMG  0(R8), R5, R6
	MOVD $0, 0(R9)          // R9 address of SAVSTACK_ASYNC
	LE_CALL                 // balr R7, R6 (return #1)
	NOPH
	MOVD R3, ret+32(FP)
	CMP  R3, $0             // compare result to 0
	BNE  done

	// retrieve errno and errno2
	MOVD  zosLibVec<>(SB), R8
	ADD   $(__errno), R8
	LMG   0(R8), R5, R6
	LE_CALL                   // balr R7, R6 __errno (return #3)
	NOPH
	MOVWZ 0(R3), R3
	MOVD  R3, err+48(FP)
	MOVD  zosLibVec<>(SB), R8
	ADD   $(__err2ad), R8
	LMG   0(R8), R5, R6
	LE_CALL                   // balr R7, R6 __err2ad (return #2)
	NOPH
	MOVW  (R3), R2            // retrieve errno2
	MOVD  R2, errno2+40(FP)   // store in return area
	XOR   R2, R2
	MOVWZ R2, (R3)            // clear errno2

done:
	MOVD R4, 0(R9)            // Save stack pointer.
	RET

//
// function to test if a pointer can be safely dereferenced (content read)
// return 0 for succces
//
TEXT ·ptrtest(SB), NOSPLIT, $0-16
	MOVD arg+0(FP), R10 // test pointer in R10

	// set up R2 to point to CEECAADMC
	BYTE $0xE3; BYTE $0x20; BYTE $0x04; BYTE $0xB8; BYTE $0x00; BYTE $0x17 // llgt  2,1208
	BYTE $0xB9; BYTE $0x17; BYTE $0x00; BYTE $0x22                         // llgtr 2,2
	BYTE $0xA5; BYTE $0x26; BYTE $0x7F; BYTE $0xFF                         // nilh  2,32767
	BYTE $0xE3; BYTE $0x22; BYTE $0x00; BYTE $0x58; BYTE $0x00; BYTE $0x04 // lg    2,88(2)
	BYTE $0xE3; BYTE $0x22; BYTE $0x00; BYTE $0x08; BYTE $0x00; BYTE $0x04 // lg    2,8(2)
	BYTE $0x41; BYTE $0x22; BYTE $0x03; BYTE $0x68                         // la    2,872(2)

	// set up R5 to point to the "shunt" path which set 1 to R3 (failure)
	BYTE $0xB9; BYTE $0x82; BYTE $0x00; BYTE $0x33 // xgr   3,3
	BYTE $0xA7; BYTE $0x55; BYTE $0x00; BYTE $0x04 // bras  5,lbl1
	BYTE $0xA7; BYTE $0x39; BYTE $0x00; BYTE $0x01 // lghi  3,1

	// if r3 is not zero (failed) then branch to finish
	BYTE $0xB9; BYTE $0x02; BYTE $0x00; BYTE $0x33 // lbl1     ltgr  3,3
	BYTE $0xA7; BYTE $0x74; BYTE $0x00; BYTE $0x08 // brc   b'0111',lbl2

	// stomic store shunt address in R5 into CEECAADMC
	BYTE $0xE3; BYTE $0x52; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x24 // stg   5,0(2)

	// now try reading from the test pointer in R10, if it fails it branches to the "lghi" instruction above
	BYTE $0xE3; BYTE $0x9A; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x04 // lg    9,0(10)

	// finish here, restore 0 into CEECAADMC
	BYTE $0xB9; BYTE $0x82; BYTE $0x00; BYTE $0x99                         // lbl2     xgr   9,9
	BYTE $0xE3; BYTE $0x92; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x24 // stg   9,0(2)
	MOVD R3, ret+8(FP)                                                     // result in R3
	RET

//
// function to test if a untptr can be loaded from a pointer
// return 1: the 8-byte content
//        2: 0 for success, 1 for failure
//
// func safeload(ptr uintptr) ( value uintptr, error uintptr)
TEXT ·safeload(SB), NOSPLIT, $0-24
	MOVD ptr+0(FP), R10                                                    // test pointer in R10
	MOVD $0x0, R6
	BYTE $0xE3; BYTE $0x20; BYTE $0x04; BYTE $0xB8; BYTE $0x00; BYTE $0x17 // llgt  2,1208
	BYTE $0xB9; BYTE $0x17; BYTE $0x00; BYTE $0x22                         // llgtr 2,2
	BYTE $0xA5; BYTE $0x26; BYTE $0x7F; BYTE $0xFF                         // nilh  2,32767
	BYTE $0xE3; BYTE $0x22; BYTE $0x00; BYTE $0x58; BYTE $0x00; BYTE $0x04 // lg    2,88(2)
	BYTE $0xE3; BYTE $0x22; BYTE $0x00; BYTE $0x08; BYTE $0x00; BYTE $0x04 // lg    2,8(2)
	BYTE $0x41; BYTE $0x22; BYTE $0x03; BYTE $0x68                         // la    2,872(2)
	BYTE $0xB9; BYTE $0x82; BYTE $0x00; BYTE $0x33                         // xgr   3,3
	BYTE $0xA7; BYTE $0x55; BYTE $0x00; BYTE $0x04                         // bras  5,lbl1
	BYTE $0xA7; BYTE $0x39; BYTE $0x00; BYTE $0x01                         // lghi  3,1
	BYTE $0xB9; BYTE $0x02; BYTE $0x00; BYTE $0x33                         // lbl1     ltgr  3,3
	BYTE $0xA7; BYTE $0x74; BYTE $0x00; BYTE $0x08                         // brc   b'0111',lbl2
	BYTE $0xE3; BYTE $0x52; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x24 // stg 5,0(2)
	BYTE $0xE3; BYTE $0x6A; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x04 // lg    6,0(10)
	BYTE $0xB9; BYTE $0x82; BYTE $0x00; BYTE $0x99                         // lbl2     xgr   9,9
	BYTE $0xE3; BYTE $0x92; BYTE $0x00; BYTE $0x00; BYTE $0x00; BYTE $0x24 // stg   9,0(2)
	MOVD R6, value+8(FP)                                                   // result in R6
	MOVD R3, error+16(FP)                                                  // error in R3
	RET
