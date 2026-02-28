// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.greenteagc

package runtime

import (
	"internal/goarch"
	"internal/runtime/gc"
	"internal/runtime/sys"
	"unsafe"
)

func (s *mspan) markBitsForIndex(objIndex uintptr) markBits {
	bytep, mask := s.gcmarkBits.bitp(objIndex)
	return markBits{bytep, mask, objIndex}
}

func (s *mspan) markBitsForBase() markBits {
	return markBits{&s.gcmarkBits.x, uint8(1), 0}
}

func tryDeferToSpanScan(p uintptr, gcw *gcWork) bool {
	return false
}

func (s *mspan) initInlineMarkBits() {
}

func (s *mspan) moveInlineMarks(to *gcBits) {
	throw("unimplemented")
}

func gcUsesSpanInlineMarkBits(_ uintptr) bool {
	return false
}

func (s *mspan) inlineMarkBits() *spanInlineMarkBits {
	return nil
}

func (s *mspan) scannedBitsForIndex(objIndex uintptr) markBits {
	throw("unimplemented")
	return markBits{}
}

type spanInlineMarkBits struct {
}

func (q *spanInlineMarkBits) tryAcquire() bool {
	return false
}

type spanQueue struct {
}

func (q *spanQueue) flush() {
}

func (q *spanQueue) empty() bool {
	return true
}

func (q *spanQueue) destroy() {
}

type spanSPMC struct {
	_       sys.NotInHeap
	allnode listNodeManual
}

func freeDeadSpanSPMCs() {
	return
}

type objptr uintptr

func (w *gcWork) tryGetSpanFast() objptr {
	return 0
}

func (w *gcWork) tryGetSpan() objptr {
	return 0
}

func (w *gcWork) tryStealSpan() objptr {
	return 0
}

func scanSpan(p objptr, gcw *gcWork) {
	throw("unimplemented")
}

type sizeClassScanStats struct {
	sparseObjsScanned uint64
}

func dumpScanStats() {
	var sparseObjsScanned uint64
	for _, stats := range memstats.lastScanStats {
		sparseObjsScanned += stats.sparseObjsScanned
	}
	print("scan: total ", sparseObjsScanned, " objs\n")
	for i, stats := range memstats.lastScanStats {
		if stats == (sizeClassScanStats{}) {
			continue
		}
		if i == 0 {
			print("scan: class L ")
		} else {
			print("scan: class ", gc.SizeClassToSize[i], "B ")
		}
		print(stats.sparseObjsScanned, " objs\n")
	}
}

func (w *gcWork) flushScanStats(dst *[gc.NumSizeClasses]sizeClassScanStats) {
	for i := range w.stats {
		dst[i].sparseObjsScanned += w.stats[i].sparseObjsScanned
	}
	clear(w.stats[:])
}

// gcMarkWorkAvailable reports whether there's any non-local work available to do.
func gcMarkWorkAvailable() bool {
	if !work.full.empty() {
		return true // global work available
	}
	if work.markrootNext.Load() < work.markrootJobs.Load() {
		return true // root scan work available
	}
	return false
}

// scanObject scans the object starting at b, adding pointers to gcw.
// b must point to the beginning of a heap object or an oblet.
// scanObject consults the GC bitmap for the pointer mask and the
// spans for the size of the object.
//
//go:nowritebarrier
func scanObject(b uintptr, gcw *gcWork) {
	// Prefetch object before we scan it.
	//
	// This will overlap fetching the beginning of the object with initial
	// setup before we start scanning the object.
	sys.Prefetch(b)

	// Find the bits for b and the size of the object at b.
	//
	// b is either the beginning of an object, in which case this
	// is the size of the object to scan, or it points to an
	// oblet, in which case we compute the size to scan below.
	s := spanOfUnchecked(b)
	n := s.elemsize
	if n == 0 {
		throw("scanObject n == 0")
	}
	if s.spanclass.noscan() {
		// Correctness-wise this is ok, but it's inefficient
		// if noscan objects reach here.
		throw("scanObject of a noscan object")
	}

	var tp typePointers
	if n > maxObletBytes {
		// Large object. Break into oblets for better
		// parallelism and lower latency.
		if b == s.base() {
			// Enqueue the other oblets to scan later.
			// Some oblets may be in b's scalar tail, but
			// these will be marked as "no more pointers",
			// so we'll drop out immediately when we go to
			// scan those.
			for oblet := b + maxObletBytes; oblet < s.base()+s.elemsize; oblet += maxObletBytes {
				if !gcw.putObjFast(oblet) {
					gcw.putObj(oblet)
				}
			}
		}

		// Compute the size of the oblet. Since this object
		// must be a large object, s.base() is the beginning
		// of the object.
		n = s.base() + s.elemsize - b
		n = min(n, maxObletBytes)
		tp = s.typePointersOfUnchecked(s.base())
		tp = tp.fastForward(b-tp.addr, b+n)
	} else {
		tp = s.typePointersOfUnchecked(b)
	}

	var scanSize uintptr
	for {
		var addr uintptr
		if tp, addr = tp.nextFast(); addr == 0 {
			if tp, addr = tp.next(b + n); addr == 0 {
				break
			}
		}

		// Keep track of farthest pointer we found, so we can
		// update heapScanWork. TODO: is there a better metric,
		// now that we can skip scalar portions pretty efficiently?
		scanSize = addr - b + goarch.PtrSize

		// Work here is duplicated in scanblock and above.
		// If you make changes here, make changes there too.
		obj := *(*uintptr)(unsafe.Pointer(addr))

		// At this point we have extracted the next potential pointer.
		// Quickly filter out nil and pointers back to the current object.
		if obj != 0 && obj-b >= n {
			// Test if obj points into the Go heap and, if so,
			// mark the object.
			//
			// Note that it's possible for findObject to
			// fail if obj points to a just-allocated heap
			// object because of a race with growing the
			// heap. In this case, we know the object was
			// just allocated and hence will be marked by
			// allocation itself.
			if !tryDeferToSpanScan(obj, gcw) {
				if obj, span, objIndex := findObject(obj, b, addr-b); obj != 0 {
					greyobject(obj, b, addr-b, span, gcw, objIndex)
				}
			}
		}
	}
	gcw.bytesMarked += uint64(n)
	gcw.heapScanWork += int64(scanSize)
	if debug.gctrace > 1 {
		gcw.stats[s.spanclass.sizeclass()].sparseObjsScanned++
	}
}
