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
// [internal/buildcfg.ParseGOEXPERIMENT].
//
// If you change this struct definition, run "go generate".
type Flags struct {
	FieldTrack        bool
	PreemptibleLoops  bool
	StaticLockRanking bool
	BoringCrypto      bool

	// Regabi is split into several sub-experiments that can be
	// enabled individually. Not all combinations work.
	// The "regabi" GOEXPERIMENT is an alias for all "working"
	// subexperiments.

	// RegabiWrappers enables ABI wrappers for calling between
	// ABI0 and ABIInternal functions. Without this, the ABIs are
	// assumed to be identical so cross-ABI calls are direct.
	RegabiWrappers bool
	// RegabiArgs enables register arguments/results in all
	// compiled Go functions.
	//
	// Requires wrappers (to do ABI translation), and reflect (so
	// reflection calls use registers).
	RegabiArgs bool

	// HeapMinimum512KiB reduces the minimum heap size to 512 KiB.
	//
	// This was originally reduced as part of PacerRedesign, but
	// has been broken out to its own experiment that is disabled
	// by default.
	HeapMinimum512KiB bool

	// Arenas causes the "arena" standard library package to be visible
	// to the outside world.
	Arenas bool

	// CgoCheck2 enables an expensive cgo rule checker.
	// When this experiment is enabled, cgo rule checks occur regardless
	// of the GODEBUG=cgocheck setting provided at runtime.
	CgoCheck2 bool

	// LoopVar changes loop semantics so that each iteration gets its own
	// copy of the iteration variable.
	LoopVar bool

	// CacheProg adds support to cmd/go to use a child process to implement
	// the build cache; see https://github.com/golang/go/issues/59719.
	CacheProg bool

	// NewInliner enables a new+improved version of the function
	// inlining phase within the Go compiler.
	NewInliner bool

	// RangeFunc enables range over func.
	RangeFunc bool

	// AliasTypeParams enables type parameters for alias types.
	// Requires that gotypesalias=1 is set with GODEBUG.
	// This flag will be removed with Go 1.25.
	AliasTypeParams bool

	// SwissMap enables the SwissTable-based map implementation.
	SwissMap bool

	// SpinbitMutex enables the new "spinbit" mutex implementation on supported
	// platforms. See https://go.dev/issue/68578.
	SpinbitMutex bool

	// SyncHashTrieMap enables the HashTrieMap sync.Map implementation.
	SyncHashTrieMap bool

	// Synctest enables the testing/synctest package.
	Synctest bool
}
