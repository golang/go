// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// This file drives -logos.mode=hints (SPEC Sec 21.2 and 21.3). The report is
// built in three steps that follow the compile: logosHintsBegin records the
// facts of the checked tree before lowering rewrites it, logosHintsLowered
// adds the lowering edits and read-only placements right after lowering, and
// LogosHintsWrite adds the escape decisions after escape analysis and writes
// the file. Plain Go packages take the same path with no lowering step.

// logosHints is the report under construction in hints mode, nil otherwise.
var logosHints *logos.Hints

// logosHintsBegin starts the report of the package that the first check
// accepted, for Go and loGo/s packages alike. The hints-mode contract
// requires the output file, so a missing one is a build error.
func logosHintsBegin(pkg *types2.Package, info *types2.Info, files []*syntax.File) {
	if base.LogosMode != "hints" {
		return
	}
	if base.LogosHintsOut == "" {
		base.ErrorfAt(src.NoXPos, 0, "logos: -logos.mode=hints needs -logos.hintsout=FILE")
		base.ErrorExit()
	}
	sources, _ := nextInputs(files)
	h := logos.NewHints(pkg.Path())
	for _, file := range files {
		if !isCgoOutput(file.Pos().RelFilename()) {
			h.AddFile(pkg, info, file, sources[file])
		}
	}
	logosHints = h
}

// logosHintsLowered adds the lowering edits and the read-only placements of a
// loGo/s package to the report.
func logosHintsLowered(files []*syntax.File, edits map[*syntax.File][]logos.Edit) {
	if logosHints == nil {
		return
	}
	for _, file := range files {
		logosHints.Lowered(file, edits[file])
		logosHints.Rodata(file)
	}
}

// LogosHintsWrite adds the escape decisions to the report and writes it.
// The compiler calls it once, after escape analysis.
func LogosHintsWrite() {
	if logosHints == nil {
		return
	}
	for _, d := range escape.LogosHintDecisions() {
		if !d.Pos.IsKnown() {
			continue
		}
		pos := base.Ctxt.OutermostPos(d.Pos)
		logosHints.AddAlloc(logos.Alloc{
			File:    pos.RelFilename(),
			Line:    int(pos.Line()),
			Col:     int(pos.Col()),
			Tier:    hintTier(d.Heap),
			Message: d.Message,
			Reason:  d.Reason,
		})
	}
	if err := logosHints.Write(base.LogosHintsOut); err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: %v", err)
		base.ErrorExit()
	}
}

// hintTier names the tier of an escape decision.
func hintTier(heap bool) string {
	if heap {
		return logos.TierHeap
	}
	return logos.TierStack
}
