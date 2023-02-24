// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build asan

package syscall

import (
	"runtime"
	"unsafe"
)

const asanenabled = true

func asanRead(addr unsafe.Pointer, len int) {
	runtime.ASanRead(addr, len)
}

func asanWrite(addr unsafe.Pointer, len int) {
	runtime.ASanWrite(addr, len)
}
