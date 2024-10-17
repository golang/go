// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fips

import _ "unsafe" // for go:linkname

// The service indicator lets users of the module query whether invoked services
// are approved. Three states are stored in a per-goroutine value by the
// runtime. The indicator starts at indicatorUnset after a reset. Invoking an
// approved service transitions to indicatorTrue. Invoking a non-approved
// service transitions to indicatorFalse, and it can't leave that state until a
// reset. The idea is that functions can "delegate" checks to inner functions,
// and if there's anything non-approved in the stack, the final result is
// negative. Finally, we expose indicatorUnset as negative to the user, so that
// we don't need to explicitly annotate fully non-approved services.

//go:linkname getIndicator
func getIndicator() uint8

//go:linkname setIndicator
func setIndicator(uint8)

const (
	indicatorUnset uint8 = iota
	indicatorFalse
	indicatorTrue
)

// ResetServiceIndicator clears the service indicator for the running goroutine.
func ResetServiceIndicator() {
	setIndicator(indicatorUnset)
}

// ServiceIndicator returns true if and only if all services invoked by this
// goroutine since the last ResetServiceIndicator call are approved.
//
// If ResetServiceIndicator was not called before by this goroutine, its return
// value is undefined.
func ServiceIndicator() bool {
	return getIndicator() == indicatorTrue
}

// RecordApproved is an internal function that records the use of an approved
// service. It does not override RecordNonApproved calls in the same span.
func RecordApproved() {
	if getIndicator() == indicatorUnset {
		setIndicator(indicatorTrue)
	}
}

// RecordNonApproved is an internal function that records the use of a
// non-approved service. It overrides any RecordApproved calls in the same span.
func RecordNonApproved() {
	setIndicator(indicatorFalse)
}
