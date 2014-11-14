// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build power64 power64le
// +build linux

package runtime

import "unsafe"

// On Power64, Linux limits the user address space to 43 bits.
// (https://www.kernel.org/doc/ols/2001/ppc64.pdf)
// In addition to the 21 bits taken from the top, we can take 3 from the
// bottom, because node must be pointer-aligned, giving a total of 24 bits
// of count.

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	return uint64(uintptr(unsafe.Pointer(node)))<<21 | uint64(cnt&(1<<24-1))
}

func lfstackUnpack(val uint64) (node *lfnode, cnt uintptr) {
	node = (*lfnode)(unsafe.Pointer(uintptr(val >> 24 << 3)))
	cnt = uintptr(val & (1<<24 - 1))
	return
}
