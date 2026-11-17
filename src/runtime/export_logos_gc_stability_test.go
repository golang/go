// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

import (
	"internal/goexperiment"
	"unsafe"
)

// LogosRegionPooledRoot reports a published region head belonging to goid
// that is also in a P's reusable chunk pool. roots counts the non-nil region
// heads observed.
//
// The world is stopped while region records and P pools are inspected.
// Region records have persistent storage. Only pool nodes are dereferenced:
// a published region head may be stale if the ownership invariant is broken.
func LogosRegionPooledRoot(goid uint64) (pooled uintptr, roots int) {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	systemstack(func() {
		forEachG(func(gp *g) {
			if gp.goid != goid {
				return
			}
			for rec := gp.logosRegions; rec != nil; rec = rec.prev {
				head := rec.chunk
				if head == nil {
					continue
				}
				roots++
				for _, pp := range allp {
					for c := pp.logosArenaPool; c != nil; c = c.next {
						if c == head {
							pooled = uintptr(unsafe.Pointer(c))
							return
						}
					}
				}
			}
		})
	})
	startTheWorld(stw)
	return
}

// LogosGCGreenTea reports whether the collector uses Green Tea.
const LogosGCGreenTea = goexperiment.GreenTeaGC

// LogosGCHeapPath classifies a live Go object's span for the graph fixture.
// p must point to an object that the caller keeps alive through this call.
func LogosGCHeapPath(p unsafe.Pointer) (heap, inline bool) {
	span := spanOfHeap(uintptr(p))
	if span == nil {
		return false, false
	}
	return true, gcUsesSpanInlineMarkBits(span.elemsize)
}
