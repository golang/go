// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !ios
// +build darwin,!ios

package unix

import "unsafe"

func ptrace(request int, pid int, addr uintptr, data uintptr) error {
	return ptrace1(request, pid, addr, data)
}

func ptracePtr(request int, pid int, addr uintptr, data unsafe.Pointer) error {
	return ptrace1Ptr(request, pid, addr, data)
}
