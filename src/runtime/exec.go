// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Runtime helpers that the runtime/executor package accesses via
// //go:linkname. Centralised here so the executor implementation
// does not need direct visibility into runtime's *g, *m, or *coro
// types; it works exclusively in unsafe.Pointer values.

package runtime

import "unsafe"

// execSetOwner sets gp.execOwner to owner. owner is opaque to the
// runtime; runtime/executor uses it to point at a per-task record.
//
//go:linkname execSetOwner
func execSetOwner(gp unsafe.Pointer, owner unsafe.Pointer) {
	(*g)(gp).execOwner = owner
}

// execGetOwner returns gp.execOwner.
//
//go:linkname execGetOwner
func execGetOwner(gp unsafe.Pointer) unsafe.Pointer {
	return (*g)(gp).execOwner
}

// execCoroG returns the *g embedded in *coro c.
//
//go:linkname execCoroG
func execCoroG(c unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer((*coro)(c).gp.ptr())
}

// execCurg returns the goroutine currently running on the calling M
// (i.e. getg().m.curg).
//
//go:linkname execCurg
func execCurg() unsafe.Pointer {
	return unsafe.Pointer(getg().m.curg)
}

// execCurM returns an opaque identifier for the calling OS thread
// (specifically, the current *m).
//
//go:linkname execCurM
func execCurM() unsafe.Pointer {
	return unsafe.Pointer(getg().m)
}

// execAddNExec atomically adjusts the global executor goroutine
// count by delta.
//
//go:linkname execAddNExec
func execAddNExec(delta int64) {
	nExec.Add(delta)
}

// execInstallHooks installs the gopark/goready interception hooks.
// runtime/executor calls this exactly once during package init.
//
//go:linkname execInstallHooks
func execInstallHooks(park, ready func(gp unsafe.Pointer)) {
	parkExecHook = park
	readyExecHook = ready
}

