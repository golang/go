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
	AlignHot              int    `help:"enable hot block alignment (currently requires -pgo)" concurrent:"ok"`
	Append                int    `help:"print information about append compilation"`
	Checkptr              int    `help:"instrument unsafe pointer conversions\n0: instrumentation disabled\n1: conversions involving unsafe.Pointer are instrumented\n2: conversions to unsafe.Pointer force heap allocation" concurrent:"ok"`
	Closure               int    `help:"print information about closure compilation"`
	Defer                 int    `help:"print information about defer compilation"`
	DisableNil            int    `help:"disable nil checks" concurrent:"ok"`
	DumpInlFuncProps      string `help:"dump function properties from inl heuristics to specified file"`
	DumpInlCallSiteScores int    `help:"dump scored callsites during inlining"`
	InlScoreAdj           string `help:"set inliner score adjustments (ex: -d=inlscoreadj=panicPathAdj:10/passConstToNestedIfAdj:-90)"`
	InlBudgetSlack        int    `help:"amount to expand the initial inline budget when new inliner enabled. Defaults to 80 if option not set." concurrent:"ok"`
	DumpPtrs              int    `help:"show Node pointers values in dump output"`
	DwarfInl              int    `help:"print information about DWARF inlined function creation"`
	EscapeMutationsCalls  int    `help:"print extra escape analysis diagnostics about mutations and calls" concurrent:"ok"`
	Export                int    `help:"print export data"`
	FIPSHash              string `help:"hash value for FIPS debugging" concurrent:"ok"`
	Fmahash               string `help:"hash value for use in debugging platform-dependent multiply-add use" concurrent:"ok"`
	GCAdjust              int    `help:"log adjustments to GOGC" concurrent:"ok"`
	GCCheck               int    `help:"check heap/gc use by compiler" concurrent:"ok"`
	GCProg                int    `help:"print dump of GC programs"`
	Gossahash             string `help:"hash value for use in debugging the compiler"`
	InlFuncsWithClosures  int    `help:"allow functions with closures to be inlined" concurrent:"ok"`
	InlStaticInit         int    `help:"allow static initialization of inlined calls" concurrent:"ok"`
	Libfuzzer             int    `help:"enable coverage instrumentation for libfuzzer"`
	LoopVar               int    `help:"shared (0, default), 1 (private loop variables), 2, private + log"`
	LoopVarHash           string `help:"for debugging changes in loop behavior. Overrides experiment and loopvar flag."`
	LocationLists         int    `help:"print information about DWARF location list creation"`
	MaxShapeLen           int    `help:"hash shape names longer than this threshold (default 500)" concurrent:"ok"`
	MergeLocals           int    `help:"merge together non-interfering local stack slots" concurrent:"ok"`
	MergeLocalsDumpFunc   string `help:"dump specified func in merge locals"`
	MergeLocalsHash       string `help:"hash value for debugging stack slot merging of local variables" concurrent:"ok"`
	MergeLocalsTrace      int    `help:"trace debug output for locals merging"`
	MergeLocalsHTrace     int    `help:"hash-selected trace debug output for locals merging"`
	Nil                   int    `help:"print information about nil checks"`
	NoDeadLocals          int    `help:"disable deadlocals pass" concurrent:"ok"`
	NoOpenDefer           int    `help:"disable open-coded defers" concurrent:"ok"`
	NoRefName             int    `help:"do not include referenced symbol names in object file" concurrent:"ok"`
	PCTab                 string `help:"print named pc-value table\nOne of: pctospadj, pctofile, pctoline, pctoinline, pctopcdata"`
	Panic                 int    `help:"show all compiler panics"`
	Reshape               int    `help:"print information about expression reshaping"`
	Shapify               int    `help:"print information about shaping recursive types"`
	Slice                 int    `help:"print information about slice compilation"`
	SoftFloat             int    `help:"force compiler to emit soft-float code" concurrent:"ok"`
	StaticCopy            int    `help:"print information about missed static copies" concurrent:"ok"`
	SyncFrames            int    `help:"how many writer stack frames to include at sync points in unified export data"`
	TailCall              int    `help:"print information about tail calls"`
	TypeAssert            int    `help:"print information about type assertion inlining"`
	WB                    int    `help:"print information about write barriers"`
	ABIWrap               int    `help:"print information about ABI wrapper generation"`
	MayMoreStack          string `help:"call named function before all stack growth checks" concurrent:"ok"`
	PGODebug              int    `help:"debug profile-guided optimizations"`
	PGOHash               string `help:"hash value for debugging profile-guided optimizations" concurrent:"ok"`
	PGOInline             int    `help:"enable profile-guided inlining" concurrent:"ok"`
	PGOInlineCDFThreshold string `help:"cumulative threshold percentage for determining call sites as hot candidates for inlining" concurrent:"ok"`
	PGOInlineBudget       int    `help:"inline budget for hot functions" concurrent:"ok"`
	PGODevirtualize       int    `help:"enable profile-guided devirtualization; 0 to disable, 1 to enable interface devirtualization, 2 to enable function devirtualization" concurrent:"ok"`
	RangeFuncCheck        int    `help:"insert code to check behavior of range iterator functions" concurrent:"ok"`
	VariableMakeHash      string `help:"hash value for debugging stack allocation of variable-sized make results" concurrent:"ok"`
	VariableMakeThreshold int    `help:"threshold in bytes for possible stack allocation of variable-sized make results" concurrent:"ok"`
	WrapGlobalMapDbg      int    `help:"debug trace output for global map init wrapping"`
	WrapGlobalMapCtl      int    `help:"global map init wrap control (0 => default, 1 => off, 2 => stress mode, no size cutoff)"`
	ZeroCopy              int    `help:"enable zero-copy string->[]byte conversions" concurrent:"ok"`

	ConcurrentOk bool // true if only concurrentOk flags seen
}

// DebugSSA is called to set a -d ssa/... option.
// If nil, those options are reported as invalid options.
// If DebugSSA returns a non-empty string, that text is reported as a compiler error.
var DebugSSA func(phase, flag string, val int, valString string) string
