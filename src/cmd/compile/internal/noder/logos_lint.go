// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"os"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// escapeLint is what the checks that read escape analysis need after it ran:
// the package the first pass checked, the position map of its findings, and
// which checks the package armed.
type escapeLint struct {
	in         *lint.Input
	m          *posMap
	pins       bool
	zeroAlloc  bool
	allocFacts bool
	graph      *logosAllocGraph
	fallback   bool
}

// logosEscape is the state logosLintHook keeps for LogosLintEscape.
var logosEscape *escapeLint

func init() {
	facts.LogosLockFacts = writeLogosLockFacts
}

// writeLogosLockFacts computes the locks records of the final checked
// package for facts.WriteChecked. It runs only when the go command
// passed a lint policy, matching the compiles the lint layer arms.
func writeLogosLockFacts(pkg *types2.Package, info *types2.Info, files []*syntax.File) []facts.LockFact {
	if base.LogosCfg == "" {
		return nil
	}
	kind := lint.KindGo
	if base.Logos {
		kind = lint.KindLgo
	}
	lintable, sources := logosLintInputs(files)
	return lint.LockFacts(&lint.Input{Pkg: pkg, Info: info, Files: lintable, Sources: sources, Kind: kind})
}

// logosLintHook runs the lints of the checked package (SPEC Sec 1.2, M9). It
// runs once after the first successful check, before lowering, so the rules
// see loGo/s syntax and types. A loGo/s package lints on every compile that
// names a policy with -logos.cfg; a Go package lints only under
// -logos.lintonly, which only logos lint passes. Findings that fail the
// compile stop it here, except under -logos.lintonly, where the compile
// continues to escape analysis so every finding of the package prints
// together.
//
// A function marked //logos:zeroalloc (SPEC Sec 17.4) is checked on the same
// compiles, without a policy: the directive is the assertion, so the check
// arms escape analysis whenever the package marks a function, and a directive
// that marks nothing fails here.
func logosLintHook(m *posMap, files []*syntax.File, pkg *types2.Package, info *types2.Info) {
	if !base.Logos && !base.LogosLintEnabled() && !base.LogosAllocFacts {
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
	var results []lint.Result
	var zeroAlloc bool
	var zeroAllocMarks []lint.ZeroAllocFunc
	if !base.LogosLintOwned {
		results, zeroAlloc, zeroAllocMarks = logosZeroAllocArm(in)
	}
	pins := false
	if logosLintActive() && !base.LogosLintOwned {
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
	var graph *logosAllocGraph
	if base.LogosAllocFacts || zeroAlloc {
		graph = logosBuildAllocGraph(in, zeroAllocMarks)
	}
	if pins || zeroAlloc || graph != nil {
		logosEscape = &escapeLint{in: in, m: m, pins: pins, zeroAlloc: zeroAlloc, allocFacts: base.LogosAllocFacts, graph: graph}
		escape.LogosPinning = pins
	}
	fallback := !base.LogosLintOwned && (base.Logos || base.LogosLintEnabled()) && lint.FallbackActive(in)
	if fallback {
		if logosEscape == nil {
			logosEscape = &escapeLint{in: in, m: m}
		}
		logosEscape.fallback = true
	}
}

// logosLintActive reports whether the compile runs the lints: a policy is
// named, and the package is loGo/s or logos lint asks for a lint compile.
func logosLintActive() bool {
	return base.LogosCfg != "" && (base.Logos || base.LogosLintEnabled() || base.LogosLintOwned)
}

// logosZeroAllocArm makes escape analysis record the heap allocation sites
// of the functions of in marked //logos:zeroalloc. It returns the results for
// directives that mark nothing, and whether any function is marked.
func logosZeroAllocArm(in *lint.Input) ([]lint.Result, bool, []lint.ZeroAllocFunc) {
	funcs, misplaced := lint.ZeroAllocFuncs(in)
	marks := make(map[escape.LogosFuncKey]string, len(funcs))
	recursive := false
	for _, f := range funcs {
		marks[escape.LogosFuncKey{File: f.File, Line: f.Line}] = f.Name
		if !f.Local {
			recursive = true
		}
	}
	escape.LogosZeroAlloc = marks
	if recursive || base.LogosAllocFacts {
		escape.LogosFuncAllocs = make(map[escape.LogosFuncKey]bool)
	}
	return misplaced, len(marks) != 0, funcs
}

// LogosLintEscape runs the checks that read escape analysis, once it has run
// over the package (SPEC Sec 1.2, M9; Sec 17.4): perf::large_backing_array_pin
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
	if st.graph != nil {
		st.graph.solve()
		if st.zeroAlloc {
			results = append(results, lint.ZeroAllocResults(st.in, st.graph.findings())...)
		}
		if st.allocFacts {
			logosMergeAllocFacts(st.graph.allocFreeObjects())
		}
	}
	if st.fallback {
		results = append(results, logosFallbackResults(st.in)...)
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

func logosFallbackResults(in *lint.Input) []lint.Result {
	found := escape.LogosRegionFallbacks()
	findings := make([]lint.FallbackFinding, 0, len(found))
	for _, f := range found {
		pos := base.Ctxt.InnermostPos(f.Pos)
		findings = append(findings, lint.FallbackFinding{
			File:    pos.RelFilename(),
			Line:    pos.Line(),
			Col:     pos.Col(),
			Message: f.Message,
		})
	}
	return lint.RunFallback(in, findings)
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
		if _, ok := stashed[f]; !ok {
			if data, err := os.ReadFile(f.Pos().RelFilename()); err == nil {
				stashed[f] = data
			}
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
		Note:         r.Note,
		OriginExpr:   r.Provenance.OriginExpr,
		OriginLine:   r.Provenance.OriginLine,
		InferredType: r.Provenance.InferredType,
	}
	for _, f := range r.Fixes {
		d.Fixes = append(d.Fixes, base.LogosFix(f))
	}
	return d
}
