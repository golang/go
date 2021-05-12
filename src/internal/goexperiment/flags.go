// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package goexperiment implements support for toolchain experiments.
//
// Toolchain experiments are controlled by the GOEXPERIMENT
// environment variable. GOEXPERIMENT is a comma-separated list of
// experiment names. GOEXPERIMENT can be set at make.bash time, which
// sets the default experiments for binaries built with the tool
// chain; or it can be set at build time. GOEXPERIMENT can also be set
// to "none", which disables any experiments that were enabled at
// make.bash time.
//
// Experiments are exposed to the build in the following ways:
//
// - Build tag goexperiment.x is set if experiment x (lower case) is
// enabled.
//
// - For each experiment x (in camel case), this package contains a
// boolean constant x and an integer constant xInt.
//
// - In runtime assembly, the macro GOEXPERIMENT_x is defined if
// experiment x (lower case) is enabled.
//
// In the toolchain, the set of experiments enabled for the current
// build should be accessed via objabi.Experiment.
//
// The set of experiments is included in the output of runtime.Version()
// and "go version <binary>" if it differs from the default experiments.
//
// For the set of experiments supported by the current toolchain, see
// "go doc goexperiment.Flags".
//
// Note that this package defines the set of experiments (in Flags)
// and records the experiments that were enabled when the package
// was compiled (as boolean and integer constants).
//
// Note especially that this package does not itself change behavior
// at run time based on the GOEXPERIMENT variable.
// The code used in builds to interpret the GOEXPERIMENT variable
// is in the separate package internal/buildcfg.
package goexperiment

//go:generate go run mkconsts.go

// Flags is the set of experiments that can be enabled or disabled in
// the current toolchain.
//
// When specified in the GOEXPERIMENT environment variable or as build
// tags, experiments use the strings.ToLower of their field name.
//
// For the baseline experimental configuration, see
// objabi.experimentBaseline.
//
// If you change this struct definition, run "go generate".
type Flags struct {
	FieldTrack        bool
	PreemptibleLoops  bool
	StaticLockRanking bool

	// Regabi is split into several sub-experiments that can be
	// enabled individually. Not all combinations work.
	// The "regabi" GOEXPERIMENT is an alias for all "working"
	// subexperiments.

	// RegabiWrappers enables ABI wrappers for calling between
	// ABI0 and ABIInternal functions. Without this, the ABIs are
	// assumed to be identical so cross-ABI calls are direct.
	RegabiWrappers bool
	// RegabiG enables dedicated G and zero registers in
	// ABIInternal.
	//
	// Requires wrappers because it makes the ABIs incompatible.
	RegabiG bool
	// RegabiReflect enables the register-passing paths in
	// reflection calls. This is also gated by intArgRegs in
	// reflect and runtime (which are disabled by default) so it
	// can be used in targeted tests.
	RegabiReflect bool
	// RegabiDefer enables desugaring defer and go calls
	// into argument-less closures.
	RegabiDefer bool
	// RegabiArgs enables register arguments/results in all
	// compiled Go functions.
	//
	// Requires wrappers (to do ABI translation), g (because
	// runtime assembly that's been ported to ABIInternal uses the
	// G register), reflect (so reflection calls use registers),
	// and defer (because the runtime doesn't support passing
	// register arguments to defer/go).
	RegabiArgs bool
}
