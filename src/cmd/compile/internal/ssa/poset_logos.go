// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

type logosPosetVisited sparseSet

func (po *poset) logosAcquireVisited() *logosPosetVisited {
	closed := po.visited
	po.visited = nil
	need := int(po.lastidx) + 1
	if closed == nil || closed.cap() < need {
		capacity := 64
		if closed != nil {
			capacity = closed.cap() * 2
		}
		for capacity < need {
			capacity *= 2
		}
		closed = newSparseSet(capacity)
	} else {
		closed.clear()
	}
	return (*logosPosetVisited)(closed)
}

func (po *poset) logosReleaseVisited(closed *logosPosetVisited) {
	po.visited = (*sparseSet)(closed)
}

func (closed *logosPosetVisited) Test(i uint32) bool {
	return (*sparseSet)(closed).contains(ID(i))
}

func (closed *logosPosetVisited) Set(i uint32) {
	(*sparseSet)(closed).add(ID(i))
}

func (closed *logosPosetVisited) Reset() {
	(*sparseSet)(closed).clear()
}
