// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9

package runtime

import "unsafe"

// logosMemAllocPerm reduces to memAlloc: mem_sbrk.go is shared with
// plan9, where the heap stays unsplit.
//
//go:systemstack
func logosMemAllocPerm(n uintptr) unsafe.Pointer {
	return memAlloc(n)
}

// logosMemFree reduces to memFree for the same reason.
//
//go:systemstack
func logosMemFree(v unsafe.Pointer, n uintptr) {
	memFree(v, n)
}
