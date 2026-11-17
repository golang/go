// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"unsafe"
)

// logosCgoPointerObject finds the allocation containing p. The caller has
// already obtained span from spanOfHeap(p), and p denotes a live allocation.
// Keep findObject's returned base and span when the experiment is disabled.
func logosCgoPointerObject(p unsafe.Pointer, span *mspan) (uintptr, *mspan) {
	if goexperiment.LogosCgoOpt {
		return span.objBase(uintptr(p)), span
	}
	base, span, _ := findObject(uintptr(p), 0, 0)
	return base, span
}
