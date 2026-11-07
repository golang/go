// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Token budget for reused goroutine stacks (GODEBUG=stackcachetokens).
//
// gfput keeps grown stacks of dead goroutines, up to maxCachedStackSize,
// and gfget gives them to new goroutines. A goroutine that exits without
// using a quarter of its reused stack spends one of the stack's tokens, and
// gfput frees the stack when the last one goes. The stack guard sits at a
// quarter of the stack; newstack refills the tokens when a goroutine hits
// it, and so does a stack move.
//
// g.stackUnused counts the spent tokens. Only the goroutine, the P that
// owns its free G, and GC while the goroutine is stopped touch it.

package runtime

// maxCachedStackSize is the largest stack that gfput keeps for reuse.
const maxCachedStackSize = 128 << 10

// stackTokenLend spends a token of gp's stack, which gfget gives to a new
// goroutine. Small stacks have no tokens: a watermark less than stackGuard
// below the top would be crossed by almost every goroutine.
func stackTokenLend(gp *g) {
	size := gp.stack.hi - gp.stack.lo
	if debug.stackcachetokens == 0 || size <= uintptr(startingStackSize) || size/4 < 2*stackGuard {
		gp.stackUnused = 0
		return
	}
	gp.stackUnused++
}

// stackTokensGone reports whether gp, which exits, spent its stack's last
// token.
func stackTokensGone(gp *g) bool {
	return gp.stackUnused != 0 && int32(gp.stackUnused) >= debug.stackcachetokens
}

// stackTokenTrap reports whether guard is the watermark of gp's stack. If
// so, it refills the tokens and restores the normal guard.
func stackTokenTrap(gp *g, guard uintptr) bool {
	if gp.stackUnused == 0 || guard != regularStackGuard(gp) {
		return false
	}
	gp.stackUnused = 0
	gp.stackguard0 = gp.stack.lo + stackGuard
	return true
}

// regularStackGuard returns gp's stack guard when no preemption is pending:
// the watermark while gp has not used a quarter of its reused stack, else
// the normal guard.
//
//go:nosplit
func regularStackGuard(gp *g) uintptr {
	if gp.stackUnused != 0 {
		return gp.stack.hi - (gp.stack.hi-gp.stack.lo)/4 + stackGuard
	}
	return gp.stack.lo + stackGuard
}
