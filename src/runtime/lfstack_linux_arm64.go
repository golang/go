// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

// In addition to the 16 bits taken from the top, we can take 3 from the
// bottom, because node must be pointer-aligned, giving a total of 19 bits
// of count.
const (
	addrBits = 48
	cntBits  = 64 - addrBits + 3
)

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	return uint64(uintptr(unsafe.Pointer(node)))<<(64-addrBits) | uint64(cnt&(1<<cntBits-1))
}

func lfstackUnpack(val uint64) (node *lfnode, cnt uintptr) {
	node = (*lfnode)(unsafe.Pointer(uintptr(val >> cntBits << 3)))
	cnt = uintptr(val & (1<<cntBits - 1))
	return
}
