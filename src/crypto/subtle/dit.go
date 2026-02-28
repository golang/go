// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle

import (
	"internal/runtime/sys"
	_ "unsafe"
)

// WithDataIndependentTiming enables architecture specific features which ensure
// that the timing of specific instructions is independent of their inputs
// before executing f. On f returning it disables these features.
//
// Any goroutine spawned by f will also have data independent timing enabled for
// its lifetime, as well as any of their descendant goroutines.
//
// Any C code called via cgo from within f, or from a goroutine spawned by f, will
// also have data independent timing enabled for the duration of the call. If the
// C code disables data independent timing, it will be re-enabled on return to Go.
//
// If C code called via cgo, from f or elsewhere, enables or disables data
// independent timing then calling into Go will preserve that state for the
// duration of the call.
//
// WithDataIndependentTiming should only be used when f is written to make use
// of constant-time operations. WithDataIndependentTiming does not make
// variable-time code constant-time.
//
// Calls to WithDataIndependentTiming may be nested.
//
// On Arm64 processors with FEAT_DIT, WithDataIndependentTiming enables
// PSTATE.DIT. See https://developer.arm.com/documentation/ka005181/1-0/?lang=en.
//
// Currently, on all other architectures WithDataIndependentTiming executes f immediately
// with no other side-effects.
//
//go:noinline
func WithDataIndependentTiming(f func()) {
	if !sys.DITSupported {
		f()
		return
	}

	alreadyEnabled := setDITEnabled()

	// disableDIT is called in a deferred function so that if f panics we will
	// still disable DIT, in case the panic is recovered further up the stack.
	defer func() {
		if !alreadyEnabled {
			setDITDisabled()
		}
	}()

	f()
}

//go:linkname setDITEnabled
func setDITEnabled() bool

//go:linkname setDITDisabled
func setDITDisabled()
