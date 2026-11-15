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

var logosHints *logos.Hints

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

func logosHintsLowered(files []*syntax.File, edits map[*syntax.File][]logos.Edit) {
	if logosHints == nil {
		return
	}
	for _, file := range files {
		logosHints.Lowered(file, edits[file])
		logosHints.Rodata(file)
	}
}

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
			Tier:    hintTier(d),
			Message: d.Message,
			Reason:  d.Reason,
		})
	}
	if err := logosHints.Write(base.LogosHintsOut); err != nil {
		base.ErrorfAt(src.NoXPos, 0, "logos: %v", err)
		base.ErrorExit()
	}
}

func hintTier(d escape.LogosHintDecision) string {
	if d.Arena {
		return logos.TierArena
	}
	if d.Heap {
		return logos.TierHeap
	}
	return logos.TierStack
}
