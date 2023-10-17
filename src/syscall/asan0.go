// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !asan

package syscall

import (
	"unsafe"
)

const asanenabled = false

func asanRead(addr unsafe.Pointer, len int) {
}

func asanWrite(addr unsafe.Pointer, len int) {
}
