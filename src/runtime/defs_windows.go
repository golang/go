// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows architecture-independent definitions.

package runtime

const (
	_PROT_NONE  = 0
	_PROT_READ  = 1
	_PROT_WRITE = 2
	_PROT_EXEC  = 4

	_MAP_ANON    = 1
	_MAP_PRIVATE = 2

	_DUPLICATE_SAME_ACCESS   = 0x2
	_THREAD_PRIORITY_HIGHEST = 0x2

	_SIGINT              = 0x2
	_SIGTERM             = 0xF
	_CTRL_C_EVENT        = 0x0
	_CTRL_BREAK_EVENT    = 0x1
	_CTRL_CLOSE_EVENT    = 0x2
	_CTRL_LOGOFF_EVENT   = 0x5
	_CTRL_SHUTDOWN_EVENT = 0x6

	_EXCEPTION_ACCESS_VIOLATION     = 0xc0000005
	_EXCEPTION_IN_PAGE_ERROR        = 0xc0000006
	_EXCEPTION_BREAKPOINT           = 0x80000003
	_EXCEPTION_ILLEGAL_INSTRUCTION  = 0xc000001d
	_EXCEPTION_FLT_DENORMAL_OPERAND = 0xc000008d
	_EXCEPTION_FLT_DIVIDE_BY_ZERO   = 0xc000008e
	_EXCEPTION_FLT_INEXACT_RESULT   = 0xc000008f
	_EXCEPTION_FLT_OVERFLOW         = 0xc0000091
	_EXCEPTION_FLT_UNDERFLOW        = 0xc0000093
	_EXCEPTION_INT_DIVIDE_BY_ZERO   = 0xc0000094
	_EXCEPTION_INT_OVERFLOW         = 0xc0000095

	_INFINITE     = 0xffffffff
	_WAIT_TIMEOUT = 0x102

	_EXCEPTION_CONTINUE_EXECUTION  = -0x1
	_EXCEPTION_CONTINUE_SEARCH     = 0x0
	_EXCEPTION_CONTINUE_SEARCH_SEH = 0x1
)

type systeminfo struct {
	anon0                       [4]byte
	dwpagesize                  uint32
	lpminimumapplicationaddress *byte
	lpmaximumapplicationaddress *byte
	dwactiveprocessormask       uintptr
	dwnumberofprocessors        uint32
	dwprocessortype             uint32
	dwallocationgranularity     uint32
	wprocessorlevel             uint16
	wprocessorrevision          uint16
}

type exceptionpointers struct {
	record  *exceptionrecord
	context *context
}

type exceptionrecord struct {
	exceptioncode        uint32
	exceptionflags       uint32
	exceptionrecord      *exceptionrecord
	exceptionaddress     uintptr
	numberparameters     uint32
	exceptioninformation [15]uintptr
}

type overlapped struct {
	internal     uintptr
	internalhigh uintptr
	anon0        [8]byte
	hevent       *byte
}

type memoryBasicInformation struct {
	baseAddress       uintptr
	allocationBase    uintptr
	allocationProtect uint32
	regionSize        uintptr
	state             uint32
	protect           uint32
	type_             uint32
}
