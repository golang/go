// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// func servicemain(argc uint32, argv **uint16)
TEXT ·servicemain(SB),7,$0
	MOVL	argc+0(FP), AX
	MOVL	AX, ·sArgc(SB)
	MOVL	argv+4(FP), AX
	MOVL	AX, ·sArgv(SB)

	PUSHL	BP
	PUSHL	BX
	PUSHL	SI
	PUSHL	DI

	SUBL	$12, SP

	MOVL	·sName(SB), AX
	MOVL	AX, (SP)
	MOVL	$·servicectlhandler(SB), AX
	MOVL	AX, 4(SP)
	MOVL	$0, 8(SP)
	MOVL	·cRegisterServiceCtrlHandlerExW(SB), AX
	MOVL	SP, BP
	CALL	AX
	MOVL	BP, SP
	CMPL	AX, $0
	JE	exit
	MOVL	AX, ·ssHandle(SB)

	MOVL	·goWaitsH(SB), AX
	MOVL	AX, (SP)
	MOVL	·cSetEvent(SB), AX
	MOVL	SP, BP
	CALL	AX
	MOVL	BP, SP

	MOVL	·cWaitsH(SB), AX
	MOVL	AX, (SP)
	MOVL	$-1, AX
	MOVL	AX, 4(SP)
	MOVL	·cWaitForSingleObject(SB), AX
	MOVL	SP, BP
	CALL	AX
	MOVL	BP, SP

exit:
	ADDL	$12, SP

	POPL	DI
	POPL	SI
	POPL	BX
	POPL	BP

	MOVL	0(SP), CX
	ADDL	$12, SP
	JMP	CX

// I do not know why, but this seems to be the only way to call
// ctlHandlerProc on Windows 7.

// func servicectlhandler(ctl uint32, evtype uint32, evdata uintptr, context uintptr) uintptr {
TEXT ·servicectlhandler(SB),7,$0
	MOVL	·ctlHandlerExProc(SB), CX
	JMP	CX
