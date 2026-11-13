// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Lend budget for reused goroutine stacks (GODEBUG=stackcachelendkb).
//
// gfput keeps grown stacks of dead goroutines, up to maxCachedStackSize,
// and gfget gives them to new goroutines. A new goroutine holds budget for
// a reused stack bigger than startingStackSize until it uses a quarter of
// the stack: its stack guard sits there, and newstack releases the budget
// when the goroutine hits it. If the budget is full, gfget frees the stack.
//
// Each P holds budget as credit in p.stackLendCredit. It takes budget from
// stackLentBytes in batches, and returns credit above two batches, at the
// end of a GC, and when it goes idle or is destroyed.

package runtime

import (
	"internal/cpu"
	"internal/runtime/atomic"
)

// maxCachedStackSize is the largest stack that gfput keeps for reuse.
const maxCachedStackSize = 128 << 10

// stackLendBatch returns the budget a P takes at once: 1/8 of its share.
func stackLendBatch() int64 {
	return int64(debug.stackcachelendkb) << 7
}

// stackLendKeep is the credit a P keeps after a release: one batch, or 0
// if the budget has no room for a batch. Every release reads it.
var stackLendKeep struct {
	_ cpu.CacheLinePad
	atomic.Int64
	_ cpu.CacheLinePad
}

// stackLentBytes is the budget in use: lent stacks plus the credit of all Ps.
var stackLentBytes atomic.Int64

// stackLend reports whether gfget may give gp's stack to a new goroutine,
// and takes budget from pp if the stack needs it.
func stackLend(pp *p, gp *g) bool {
	size := gp.stack.hi - gp.stack.lo
	if debug.gcshrinkstackoff > 0 {
		// GC could never shrink a reused stack.
		return size == uintptr(startingStackSize)
	}
	if debug.stackcachelendkb == 0 || size <= uintptr(startingStackSize) || size/4 <= stackGuard {
		return true
	}
	s := int64(size)
	if pp.stackLendCredit < s {
		// Take a batch, or only s if a batch does not fit. Load before
		// Add, so a full budget costs no atomic write.
		limit := int64(gomaxprocs) * (int64(debug.stackcachelendkb) << 10)
		keep := stackLendBatch()
		n := max(s, keep) - pp.stackLendCredit
		used := stackLentBytes.Load()
		if used+n > limit {
			keep = 0
			n = s - pp.stackLendCredit
		}
		if stackLendKeep.Load() != keep {
			stackLendKeep.Store(keep)
		}
		if used+n > limit {
			return false
		}
		if stackLentBytes.Add(n) > limit {
			stackLentBytes.Add(-n)
			return false
		}
		pp.stackLendCredit += n
	}
	pp.stackLendCredit -= s
	gp.stackLent = true
	return true
}

// stackLendRelease returns gp's budget, if any, to pp. It must run before
// gp's stack moves and when gp exits.
func stackLendRelease(pp *p, gp *g) {
	if !gp.stackLent {
		return
	}
	gp.stackLent = false
	s := int64(gp.stack.hi - gp.stack.lo)
	if pp == nil {
		stackLentBytes.Add(-s)
		return
	}
	pp.stackLendCredit += s
	if keep := stackLendKeep.Load(); pp.stackLendCredit > 2*keep {
		stackLentBytes.Add(keep - pp.stackLendCredit)
		pp.stackLendCredit = keep
	}
}

// stackLendFlush returns all of pp's credit. The caller owns pp.
func stackLendFlush(pp *p) {
	if pp.stackLendCredit != 0 {
		stackLentBytes.Add(-pp.stackLendCredit)
		pp.stackLendCredit = 0
	}
}

// stackLendTrap reports whether guard is the watermark of gp's lent stack.
// If so, it releases gp's budget and restores the normal guard.
func stackLendTrap(pp *p, gp *g, guard uintptr) bool {
	if !gp.stackLent || guard != regularStackGuard(gp) {
		return false
	}
	stackLendRelease(pp, gp)
	gp.stackguard0 = gp.stack.lo + stackGuard
	return true
}

// regularStackGuard returns gp's stack guard when no preemption is pending:
// the watermark while gp holds budget, else the normal guard.
//
//go:nosplit
func regularStackGuard(gp *g) uintptr {
	if gp.stackLent {
		return gp.stack.hi - (gp.stack.hi-gp.stack.lo)/4 + stackGuard
	}
	return gp.stack.lo + stackGuard
}
