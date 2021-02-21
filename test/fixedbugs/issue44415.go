// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package p

import (
	"syscall"
	"unsafe"
)

var dllKernel = syscall.NewLazyDLL("Kernel32.dll")

func Call() {
	procLocalFree := dllKernel.NewProc("LocalFree")
	defer procLocalFree.Call(uintptr(unsafe.Pointer(nil)))
}
