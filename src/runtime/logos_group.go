// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe"

// This file implements the per-goroutine group tag backing
// logos/rt.Group. Each goroutine carries the ID of the loGo/s group it
// runs for in its g field (0 outside a group). Group.Go tags its child
// at start through logosSetGroup and clears the tag on exit, so a
// recycled g never keeps a stale ID: every tag is set before use, and
// the deferred clear runs on every exit path (return, recovered panic,
// and Goexit runs deferred calls).
//
// logosSetGroup is referenced from logos/rt via //go:linkname pull and
// carries a push-style //go:linkname mark here, so the linker's linkname
// check accepts the pull. The netpoll wake side (registry, cancel) lands
// in M8-S4; context cancellation alone carries S1.

//go:linkname logosSetGroup
func logosSetGroup(id uint64) {
	getg().logosGroup = id
}
