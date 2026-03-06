// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package windows provides the syscall primitives required for the runtime.

package windows

import (
	"internal/abi"
)

// MaxArgs should be divisible by 2, as Windows stack
// must be kept 16-byte aligned on syscall entry.
//
// Although it only permits maximum 42 parameters, it
// is arguably large enough.
const MaxArgs = 42

// StdCallInfo is a structure used to pass parameters to the system call.
type StdCallInfo struct {
	Fn   uintptr
	N    uintptr // number of parameters
	Args uintptr // parameters
	R1   uintptr // return values
	R2   uintptr
}

// StdCall calls a function using Windows' stdcall convention.
// The calling thread's last-error code value is cleared before calling the function,
// and stored in the return value.
//
//go:noescape
func StdCall(fn *StdCallInfo) uint32

// asmstdcall is the function pointer for [AsmStdCallAddr].
// The calling thread's last-error code value is cleared before calling the function,
// and returned in the C ABI return register (not via Go stack convention).
// This function is not called directly from Go; it is either jumped to from
// [StdCall] or called from C via [AsmStdCallAddr].
func asmstdcall(fn *StdCallInfo)

// AsmStdCallAddr is the address of a function that accepts a pointer
// to [StdCallInfo] stored on the stack following the C calling convention,
// and calls the function using Windows' stdcall calling convention.
// Shouldn't be called directly from Go.
func AsmStdCallAddr() uintptr {
	return abi.FuncPCABI0(asmstdcall)
}
