// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ios
// +build ios

package unix

import "unsafe"

func ptrace(request int, pid int, addr uintptr, data uintptr) (err error) {
	return ENOTSUP
}

func ptracePtr(request int, pid int, addr uintptr, data unsafe.Pointer) (err error) {
	return ENOTSUP
}
