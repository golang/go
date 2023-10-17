// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build msan

package syscall

import (
	"runtime"
	"unsafe"
)

const msanenabled = true

func msanRead(addr unsafe.Pointer, len int) {
	runtime.MSanRead(addr, len)
}

func msanWrite(addr unsafe.Pointer, len int) {
	runtime.MSanWrite(addr, len)
}
