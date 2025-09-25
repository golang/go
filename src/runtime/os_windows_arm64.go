// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:nosplit
func cputicks() int64 {
	var counter int64
	stdcall(_QueryPerformanceCounter, uintptr(unsafe.Pointer(&counter)))
	return counter
}

func stackcheck() {
	// TODO: not implemented
}
