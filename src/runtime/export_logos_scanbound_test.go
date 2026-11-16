// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.logosarena

package runtime

// LogosArenaLeaseScanBounds samples the root scan bound through an arena
// chunk's lease, pool reuse, and release. It also forces one failed lease.
// It drains only the current P's pool of empty chunks. The world stays
// stopped so physical stack changes cannot affect the observed deltas.
func LogosArenaLeaseScanBounds() (bounds [7]int64, reused, rejected, ok bool) {
	stw := stopTheWorld(stwForTestCountPagesInUse)
	systemstack(func() {
		pp := getg().m.p.ptr()
		bound := func() int64 {
			n := int64(gcController.maxStackScan.Load())
			for _, p := range allp {
				n += p.maxStackScanDelta
			}
			return n
		}

		logosArenaFlushP(pp)
		bounds[0] = bound()
		if c := logosArenaTake(); c != nil {
			bounds[1] = bound()
			logosArenaRelease(c)
			bounds[2] = bound()
			if next := logosArenaTake(); next != nil {
				reused = next == c
				bounds[3] = bound()
				logosArenaRelease(next)
				bounds[4] = bound()
				ok = true
			}
		}
		logosArenaFlushP(pp)
		bounds[5] = bound()

		// With no pooled chunk and no room for another allocation, Take
		// must fail. Restore the limit and reclaim any unexpected lease
		// before the world can resume, even if this check detects a bug.
		limit := logosArenaChunkLimit.Swap(logosArenaChunkTotal.Load())
		c := logosArenaTake()
		logosArenaChunkLimit.Store(limit)
		rejected = c == nil
		if c != nil {
			logosArenaRelease(c)
			logosArenaFlushP(pp)
		}
		bounds[6] = bound()
	})
	startTheWorld(stw)
	return
}
