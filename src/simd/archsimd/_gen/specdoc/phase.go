// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package specdoc

import (
	"simd/archsimd/_gen/sgutil"
	"simd/archsimd/_gen/specgen"
	"simd/archsimd/_gen/specgen/specexpr"
)

// detectUnknown implements [Options.RejectUnknown] for the exported
// declaration decl, with the matching spec function specFn or spec type
// specType (or neither). It reports decl if it doesn't belong in the file,
// either because it's unknown or because it isn't allowed in a file of the
// file's phase.
func (f *filler) detectUnknown(decl Decl, specFn *specgen.Func, specType *specgen.TypeDecl) {
	if !f.opts.RejectUnknown || f.test {
		return
	}
	allowed, apiReason := f.allowedPhases(decl, specFn, specType)
	if allowed.has(f.phase) {
		return
	}
	if apiReason == "" {
		// decl is in neither spec nor sgutil.NonSpecAPI, and the file isn't
		// experimental, so report it as unknown.
		f.report.UnknownDecls = append(f.report.UnknownDecls, UnknownDecl{decl})
		return
	}
	var fileReason string
	switch f.phase {
	case 0:
		fileReason = "the file builds without GOEXPERIMENT=simdstable or simd"
	case sgutil.PhaseStable:
		fileReason = "the file is generally available (goexperiment.simdstable)"
	case sgutil.PhaseExperiment:
		fileReason = "the file is experimental (goexperiment.simd)"
	}
	f.report.PhaseViolations = append(f.report.PhaseViolations, PhaseViolation{D: decl, Phase: f.phase, Reason: apiReason + ", but " + fileReason})
}

// A phaseSet is a set of [sgutil.Phase]s.
type phaseSet uint8

// phases returns the set of ps.
func phases(ps ...sgutil.Phase) phaseSet {
	var s phaseSet
	for _, p := range ps {
		s |= 1 << p
	}
	return s
}

// has reports whether s contains p.
func (s phaseSet) has(p sgutil.Phase) bool { return s&(1<<p) != 0 }

// allowedPhases returns the phases of the files that may declare decl, with
// the matching spec function specFn or spec type specType (or neither), and
// the reason decl is allowed only in those phases. If a file's phase isn't one
// of them, decl doesn't belong in the file, and the reason is used to report
// that. The reason is "" if decl is unknown (in neither spec nor
// [sgutil.NonSpecAPI]), which is reported separately.
//
// No file without a phase may declare anything, since it builds regardless of
// GOEXPERIMENT. The APIs in [sgutil.NonSpecAPI] may be in a file of either
// phase. Experimental APIs, including unknown ones, may only be in an
// experimental file, and generally available APIs only in a generally
// available one.
//
// An API described by spec is generally available unless spec marks it
// //specgen:experiment, except that in archsimd, an API is only generally
// available if it uses only fixed-width vectors, and only on the
// architectures whose archsimd API can be generally available (see
// [sgutil.BuildConstraints.StableArchOnly]). Everything else in archsimd is
// experimental until its architecture is added to that list. This matters
// because spec describes vector shapes, not packages: for example, a
// scalable shape such as Int8s describes both the portable simd.Int8s and
// archsimd.Int8s, the experimental SVE type.
func (f *filler) allowedPhases(decl Decl, specFn *specgen.Func, specType *specgen.TypeDecl) (allowed phaseSet, reason string) {
	stable, experimental := phases(sgutil.PhaseStable), phases(sgutil.PhaseExperiment)

	// The SIMD APIs that spec doesn't describe, such as the CPU feature
	// checks, may be in a file of either phase.
	if sgutil.NonSpecAPI(f.pkg, decl.Recv, decl.Name) {
		return stable | experimental, "listed in sgutil.NonSpecAPI"
	}

	// Anything else that spec doesn't describe is unknown. Experimental APIs
	// needn't be in spec, so unknown APIs may be in an experimental file.
	if specFn == nil && specType == nil {
		return experimental, ""
	}

	// The rest are described by spec, and are generally available unless
	// one of these makes them experimental.
	if specFn != nil && specFn.Experiment {
		return experimental, "marked //specgen:experiment in spec"
	}
	if f.pkg == "archsimd" {
		if specType != nil && isScalable(specType.Shape) || specFn != nil && hasScalable(specFn) {
			return experimental, "archsimd API with scalable vectors, which is experimental"
		}
		if !f.stableArchOnly {
			return experimental, "archsimd API in a file that can build for architectures other than amd64, arm64 and wasm"
		}
	}
	return stable, "generally available in spec"
}

// isScalable reports whether t is a vector type with a scalable width.
func isScalable(t specexpr.Type) bool {
	v, ok := t.(specexpr.Vector)
	if !ok {
		return false
	}
	_, fixed := v.Width.(specexpr.Int)
	return !fixed
}

// hasScalable reports whether fn's receiver, parameters or results have
// scalable vector types.
func hasScalable(fn *specgen.Func) bool {
	if fn.Recv.Type != nil && isScalable(fn.Recv.Type) {
		return true
	}
	for _, args := range [][]specgen.Arg{fn.In, fn.Out} {
		for _, a := range args {
			if isScalable(a.Type) {
				return true
			}
		}
	}
	return false
}
