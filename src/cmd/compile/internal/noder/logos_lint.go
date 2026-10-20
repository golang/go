// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"os"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// escapeLint is what the checks that read escape analysis need after it ran:
// the package the first pass checked, the position map of its findings, and
// which checks the package armed.
type escapeLint struct {
	in        *lint.Input
	m         *posMap
	pins      bool
	zeroAlloc bool
}

// logosEscape is the state logosLintHook keeps for LogosLintEscape.
var logosEscape *escapeLint

// logosLintHook runs the lints of the checked package (SPEC Sec 1.2, M9). It
// runs once after the first successful check, before lowering, so the rules
// see loGo/s syntax and types. A loGo/s package lints on every compile that
// names a policy with -logos.cfg; a Go package lints only under
// -logos.lintonly, which only logos lint passes. Findings that fail the
// compile stop it here, except under -logos.lintonly, where the compile
// continues to escape analysis so every finding of the package prints
// together.
//
// A function marked //logos:zeroalloc (SPEC Sec 22.4) is checked on the same
// compiles, without a policy: the directive is the assertion, so the check
// arms escape analysis whenever the package marks a function, and a directive
// that marks nothing fails here.
func logosLintHook(m *posMap, files []*syntax.File, pkg *types2.Package, info *types2.Info) {
	if !base.Logos && !base.LogosLintOnly {
		return
	}
	kind := lint.KindGo
	if base.Logos {
		kind = lint.KindLgo
	}
	lintable, sources := logosLintInputs(files)
	in := &lint.Input{
		Pkg:     pkg,
		Info:    info,
		Files:   lintable,
		Sources: sources,
		Kind:    kind,
	}
	results, zeroAlloc := logosZeroAllocArm(in)
	pins := false
	if base.LogosCfg != "" {
		in.Policy = logosLintPolicy()
		in.Reported = logosTakeSoft()
		results = append(results, lint.Run(in)...)
		pins = lint.PinActive(in)
	}
	for _, r := range results {
		base.LogosReport(logosDiag(m, r))
	}
	if !base.LogosLintOnly {
		base.ExitIfErrors()
	}
	if pins || zeroAlloc {
		logosEscape = &escapeLint{in: in, m: m, pins: pins, zeroAlloc: zeroAlloc}
		escape.LogosPinning = pins
	}
}

// logosLintActive reports whether the compile runs the lints: a policy is
// named, and the package is loGo/s or logos lint asks for a lint compile.
func logosLintActive() bool {
	return base.LogosCfg != "" && (base.Logos || base.LogosLintOnly)
}

// logosZeroAllocArm makes escape analysis record the heap allocation sites
// of the functions of in marked //logos:zeroalloc. It returns the results for
// directives that mark nothing, and whether any function is marked.
func logosZeroAllocArm(in *lint.Input) ([]lint.Result, bool) {
	funcs, misplaced := lint.ZeroAllocFuncs(in)
	marks := make(map[escape.LogosFuncKey]string, len(funcs))
	for _, f := range funcs {
		marks[escape.LogosFuncKey{File: f.File, Line: f.Line}] = f.Name
	}
	escape.LogosZeroAlloc = marks
	return misplaced, len(marks) != 0
}

// LogosLintEscape runs the checks that read escape analysis, once it has run
// over the package (SPEC Sec 1.2, M9; Sec 22.4): perf::large_backing_array_pin
// and perf::zeroalloc_violation. It reports findings like logosLintHook and
// stops the compile on an error, except under -logos.lintonly, where the
// compile ends right after.
func LogosLintEscape() {
	st := logosEscape
	if st == nil {
		return
	}
	var results []lint.Result
	if st.pins {
		results = append(results, logosPinResults(st.in)...)
	}
	if st.zeroAlloc {
		results = append(results, logosZeroAllocResults(st.in)...)
	}
	for _, r := range results {
		base.LogosReport(logosDiag(st.m, r))
	}
	if !base.LogosLintOnly {
		base.ExitIfErrors()
	}
}

// logosZeroAllocResults converts the allocation sites that escape analysis
// recorded inside marked functions to results. A site inside an inlined body
// belongs to the call that inlined it, so it is reported at that call.
func logosZeroAllocResults(in *lint.Input) []lint.Result {
	allocs := escape.LogosAllocs()
	found := make([]lint.ZeroAllocFinding, 0, len(allocs))
	for _, a := range allocs {
		pos := base.Ctxt.OutermostPos(a.Pos)
		found = append(found, lint.ZeroAllocFinding{
			Func:   a.Func,
			File:   pos.RelFilename(),
			Line:   pos.Line(),
			Col:    pos.Col(),
			Reason: a.Reason,
		})
	}
	return lint.ZeroAllocResults(in, found)
}

// logosPinResults resolves the large backing store pins that escape analysis
// found to results.
func logosPinResults(in *lint.Input) []lint.Result {
	pins := escape.LogosPins()
	found := make([]lint.PinFinding, 0, len(pins))
	for _, p := range pins {
		pos := base.Ctxt.InnermostPos(p.Pos)
		found = append(found, lint.PinFinding{
			File:      pos.RelFilename(),
			Line:      pos.Line(),
			Col:       pos.Col(),
			Bytes:     p.Bytes,
			Substring: p.Substring,
		})
	}
	return lint.RunPins(in, found)
}

// logosLintPolicy reads the -logos.cfg policy. A policy the go command
// wrote and this compiler cannot read is a build error.
func logosLintPolicy() *lint.Policy {
	data, err := os.ReadFile(base.LogosCfg)
	if err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: %v", err)
		base.ErrorExit()
	}
	policy, err := lint.ParsePolicy(data)
	if err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: -logos.cfg %s: %v", base.LogosCfg, err)
		base.ErrorExit()
	}
	return policy
}

// logosLintInputs returns the files to lint, without compiler-generated cgo
// outputs, and the original bytes the parser stashed for them.
func logosLintInputs(files []*syntax.File) ([]*syntax.File, map[*syntax.File][]byte) {
	stashed, _ := nextInputs(files)
	var lintable []*syntax.File
	for _, f := range files {
		if !isCgoOutput(f.Pos().RelFilename()) {
			lintable = append(lintable, f)
		}
	}
	return lintable, stashed
}

// logosDiag converts a lint result to the compiler diagnostic sink.
func logosDiag(m *posMap, r lint.Result) base.LogosDiag {
	d := base.LogosDiag{
		Rule:         r.Rule,
		Severity:     r.Severity.String(),
		Pos:          m.makeXPos(r.Pos),
		End:          m.makeXPos(r.End),
		Message:      r.Message,
		OriginExpr:   r.Provenance.OriginExpr,
		OriginLine:   r.Provenance.OriginLine,
		InferredType: r.Provenance.InferredType,
	}
	for _, f := range r.Fixes {
		d.Fixes = append(d.Fixes, base.LogosFix(f))
	}
	return d
}
