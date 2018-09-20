// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

package runtime

import "unsafe"

const (
	debugCallSystemStack = "executing on Go runtime stack"
	debugCallUnknownFunc = "call from unknown function"
	debugCallRuntime     = "call from within the Go runtime"
	debugCallUnsafePoint = "call not at safe point"
)

func debugCallV1()
func debugCallPanicked(val interface{})

// debugCallCheck checks whether it is safe to inject a debugger
// function call with return PC pc. If not, it returns a string
// explaining why.
//
//go:nosplit
func debugCallCheck(pc uintptr) string {
	// No user calls from the system stack.
	if getg() != getg().m.curg {
		return debugCallSystemStack
	}
	if sp := getcallersp(); !(getg().stack.lo < sp && sp <= getg().stack.hi) {
		// Fast syscalls (nanotime) and racecall switch to the
		// g0 stack without switching g. We can't safely make
		// a call in this state. (We can't even safely
		// systemstack.)
		return debugCallSystemStack
	}

	// Switch to the system stack to avoid overflowing the user
	// stack.
	var ret string
	systemstack(func() {
		f := findfunc(pc)
		if !f.valid() {
			ret = debugCallUnknownFunc
			return
		}

		// Disallow calls from the runtime. We could
		// potentially make this condition tighter (e.g., not
		// when locks are held), but there are enough tightly
		// coded sequences (e.g., defer handling) that it's
		// better to play it safe.
		if name, pfx := funcname(f), "runtime."; len(name) > len(pfx) && name[:len(pfx)] == pfx {
			ret = debugCallRuntime
			return
		}

		// Look up PC's register map.
		pcdata := int32(-1)
		if pc != f.entry {
			pc--
			pcdata = pcdatavalue(f, _PCDATA_RegMapIndex, pc, nil)
		}
		if pcdata == -1 {
			pcdata = 0 // in prologue
		}
		stkmap := (*stackmap)(funcdata(f, _FUNCDATA_RegPointerMaps))
		if pcdata == -2 || stkmap == nil {
			// Not at a safe point.
			ret = debugCallUnsafePoint
			return
		}
	})
	return ret
}

// debugCallWrap pushes a defer to recover from panics in debug calls
// and then calls the dispatching function at PC dispatch.
func debugCallWrap(dispatch uintptr) {
	var dispatchF func()
	dispatchFV := funcval{dispatch}
	*(*unsafe.Pointer)(unsafe.Pointer(&dispatchF)) = noescape(unsafe.Pointer(&dispatchFV))

	var ok bool
	defer func() {
		if !ok {
			err := recover()
			debugCallPanicked(err)
		}
	}()
	dispatchF()
	ok = true
}
