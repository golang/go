// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"syscall"
	"unsafe"
)

func newIovecWithBase(base *byte) syscall.Iovec {
	return syscall.Iovec{Base: (*int8)(unsafe.Pointer(base))}
}
