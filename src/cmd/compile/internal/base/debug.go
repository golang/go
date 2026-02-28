// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Debug arguments, set by -d flag.

package base

// Debug holds the parsed debugging configuration values.
var Debug DebugFlags

// DebugFlags defines the debugging configuration values (see var Debug).
// Each struct field is a different value, named for the lower-case of the field name.
// Each field must be an int or string and must have a `help` struct tag.
//
// The -d option takes a comma-separated list of settings.
// Each setting is name=value; for ints, name is short for name=1.
type DebugFlags struct {
	Append               int    `help:"print information about append compilation"`
	Checkptr             int    `help:"instrument unsafe pointer conversions\n0: instrumentation disabled\n1: conversions involving unsafe.Pointer are instrumented\n2: conversions to unsafe.Pointer force heap allocation"`
	Closure              int    `help:"print information about closure compilation"`
	DclStack             int    `help:"run internal dclstack check"`
	Defer                int    `help:"print information about defer compilation"`
	DisableNil           int    `help:"disable nil checks"`
	DumpPtrs             int    `help:"show Node pointers values in dump output"`
	DwarfInl             int    `help:"print information about DWARF inlined function creation"`
	Export               int    `help:"print export data"`
	GCProg               int    `help:"print dump of GC programs"`
	InlFuncsWithClosures int    `help:"allow functions with closures to be inlined"`
	Libfuzzer            int    `help:"enable coverage instrumentation for libfuzzer"`
	LocationLists        int    `help:"print information about DWARF location list creation"`
	Nil                  int    `help:"print information about nil checks"`
	NoOpenDefer          int    `help:"disable open-coded defers"`
	PCTab                string `help:"print named pc-value table\nOne of: pctospadj, pctofile, pctoline, pctoinline, pctopcdata"`
	Panic                int    `help:"show all compiler panics"`
	Slice                int    `help:"print information about slice compilation"`
	SoftFloat            int    `help:"force compiler to emit soft-float code"`
	SyncFrames           int    `help:"how many writer stack frames to include at sync points in unified export data"`
	TypeAssert           int    `help:"print information about type assertion inlining"`
	TypecheckInl         int    `help:"eager typechecking of inline function bodies"`
	Unified              int    `help:"enable unified IR construction"`
	WB                   int    `help:"print information about write barriers"`
	ABIWrap              int    `help:"print information about ABI wrapper generation"`
	MayMoreStack         string `help:"call named function before all stack growth checks"`

	Any bool // set when any of the debug flags have been set
}

// DebugSSA is called to set a -d ssa/... option.
// If nil, those options are reported as invalid options.
// If DebugSSA returns a non-empty string, that text is reported as a compiler error.
var DebugSSA func(phase, flag string, val int, valString string) string
