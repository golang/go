// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

type stdFunction *byte

func stdcall0(fn stdFunction) uintptr
func stdcall1(fn stdFunction, a0 uintptr) uintptr
func stdcall2(fn stdFunction, a0, a1 uintptr) uintptr
func stdcall3(fn stdFunction, a0, a1, a2 uintptr) uintptr
func stdcall4(fn stdFunction, a0, a1, a2, a3 uintptr) uintptr
func stdcall5(fn stdFunction, a0, a1, a2, a3, a4 uintptr) uintptr
func stdcall6(fn stdFunction, a0, a1, a2, a3, a4, a5 uintptr) uintptr
func stdcall7(fn stdFunction, a0, a1, a2, a3, a4, a5, a6 uintptr) uintptr

func asmstdcall(fn unsafe.Pointer)
func getlasterror() uint32
func setlasterror(err uint32)
func usleep1(usec uint32)

const stackSystem = 512 * ptrSize
