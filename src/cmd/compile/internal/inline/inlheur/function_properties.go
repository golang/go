// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inlheur

// This file defines a set of Go function "properties" intended to
// guide inlining heuristics; these properties may apply to the
// function as a whole, or to one or more function return values or
// parameters.
//
// IMPORTANT: function properties are produced on a "best effort"
// basis, meaning that the code that computes them doesn't verify that
// the properties are guaranteed to be true in 100% of cases. For this
// reason, properties should only be used to drive always-safe
// optimization decisions (e.g. "should I inline this call", or
// "should I unroll this loop") as opposed to potentially unsafe IR
// alterations that could change program semantics (e.g. "can I delete
// this variable" or "can I move this statement to a new location").
//
//----------------------------------------------------------------

// FuncProps describes a set of function or method properties that may
// be useful for inlining heuristics. Here 'Flags' are properties that
// we think apply to the entire function; 'RecvrParamFlags' are
// properties of specific function params (or the receiver), and
// 'ResultFlags' are things properties we think will apply to values
// of specific results. Note that 'ParamFlags' includes and entry for
// the receiver if applicable, and does include etries for blank
// params; for a function such as "func foo(_ int, b byte, _ float32)"
// the length of ParamFlags will be 3.
type FuncProps struct {
	Flags       FuncPropBits
	ParamFlags  []ParamPropBits // slot 0 receiver if applicable
	ResultFlags []ResultPropBits
}

type FuncPropBits uint32

const (
	// Function always panics or invokes os.Exit() or a func that does
	// likewise.
	FuncPropNeverReturns FuncPropBits = 1 << iota
)

type ParamPropBits uint32

const (
	// No info about this param
	ParamNoInfo ParamPropBits = 0

	// Parameter value feeds unmodified into a top-level interface
	// call (this assumes the parameter is of interface type).
	ParamFeedsInterfaceMethodCall ParamPropBits = 1 << iota

	// Parameter value feeds unmodified into an interface call that
	// may be conditional/nested and not always executed (this assumes
	// the parameter is of interface type).
	ParamMayFeedInterfaceMethodCall ParamPropBits = 1 << iota

	// Parameter value feeds unmodified into a top level indirect
	// function call (assumes parameter is of function type).
	ParamFeedsIndirectCall

	// Parameter value feeds unmodified into an indirect function call
	// that is conditional/nested (not guaranteed to execute). Assumes
	// parameter is of function type.
	ParamMayFeedIndirectCall

	// Parameter value feeds unmodified into a top level "switch"
	// statement or "if" statement simple expressions (see more on
	// "simple" expression classification below).
	ParamFeedsIfOrSwitch

	// Parameter value feeds unmodified into a "switch" or "if"
	// statement simple expressions (see more on "simple" expression
	// classification below), where the if/switch is
	// conditional/nested.
	ParamMayFeedIfOrSwitch
)

type ResultPropBits uint32

const (
	// No info about this result
	ResultNoInfo ResultPropBits = 0
	// This result always contains allocated memory.
	ResultIsAllocatedMem ResultPropBits = 1 << iota
	// This result is always a single concrete type that is
	// implicitly converted to interface.
	ResultIsConcreteTypeConvertedToInterface
	// Result is always the same non-composite compile time constant.
	ResultAlwaysSameConstant
	// Result is always the same function or closure.
	ResultAlwaysSameFunc
	// Result is always the same (potentially) inlinable function or closure.
	ResultAlwaysSameInlinableFunc
)
