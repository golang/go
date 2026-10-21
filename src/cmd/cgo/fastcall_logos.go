// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"internal/buildcfg"
	"strings"
)

// Fast-cgo routing for M20-S1.
//
// A C function takes the fast call path when it carries the directive
// pair #cgo noescape F and #cgo nocallback F, or the .lgo spelling
// //logos:cgo_fastcall F in a loGo/s cgo input (which maps to the same
// pair). cmd/cgo then emits a call to runtime.logosCgocallFast instead
// of runtime.cgocall; the generated file names and the rest of the
// pipeline do not change. Go code needs no new directive: the pair is
// the trigger there.

// logosFastCall reports whether the C function n takes the fast path:
// both directives present, on the gc toolchain. gccgo keeps the plain
// entry: it links the prolog elsewhere and has no fast entry.
func logosFastCall(p *Package, n *Name) bool {
	if *gccgo {
		return false
	}
	return p.noEscapes[n.C] && p.noCallbacks[n.C]
}

// logosStableFastCall reports whether n's result wrapper may keep its
// argument frame at a fixed address. The experiment gates the existing
// pair-marked fast call; syscallsp prevents stack shrinking while C runs,
// and the trusted nocallback contract excludes callbacks that may grow the
// stack.
func logosStableFastCall(p *Package, n *Name) bool {
	return buildcfg.Experiment.LogosFastCgo && logosFastCall(p, n)
}

// logosCgoCall names the runtime entry the wrapper calls for n: the
// fast entry for pair-marked functions, the plain entry otherwise.
func logosCgoCall(p *Package, n *Name) string {
	if logosFastCall(p, n) {
		return "_cgo_runtime_cgocall_fast"
	}
	return "_cgo_runtime_cgocall"
}

// logosFastcallMarker is the .lgo spelling of the directive pair. It
// takes the C function name as its single argument:
//
//	//logos:cgo_fastcall F
//
// Like //go: directives, the marker must not contain a space after the
// slashes. Any other shape is an ordinary comment.
const logosFastcallMarker = "//logos:cgo_fastcall"

// logosMapFastcallDirectives maps //logos:cgo_fastcall lines in the
// preamble of a loGo/s cgo input into the noescape plus nocallback
// pair. It blanks consumed lines so they do not leak into the exported
// header, mirroring ProcessCgoDirectives. In .go inputs the marker is
// an ordinary comment: Go code uses the #cgo pair and gets no new
// directive. Unmatched names flow into the existing pair validation,
// which reports the same no-matched-C-function error.
func logosMapFastcallDirectives(f *File, srcfile string) {
	if !isLogosCgoInput(srcfile) {
		return
	}
	if f.NoCallbacks == nil {
		f.NoCallbacks = make(map[string]bool)
	}
	if f.NoEscapes == nil {
		f.NoEscapes = make(map[string]bool)
	}
	lines := strings.Split(f.Preamble, "\n")
	for i, line := range lines {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != logosFastcallMarker {
			continue
		}
		f.NoCallbacks[fields[1]] = true
		f.NoEscapes[fields[1]] = true
		lines[i] = ""
	}
	f.Preamble = strings.Join(lines, "\n")
}
