// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386 arm mips mipsle

package runtime

import "unsafe"

// On 32-bit systems, the stored uint64 has a 32-bit pointer and 32-bit count.

func lfstackPack(node *lfnode, cnt uintptr) uint64 {
	return uint64(uintptr(unsafe.Pointer(node)))<<32 | uint64(cnt)
}

func lfstackUnpack(val uint64) *lfnode {
	return (*lfnode)(unsafe.Pointer(uintptr(val >> 32)))
}
