// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"internal/logosfile"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/objabi"
)

// logosCgoSourceName recovers an overlay's logical source identity only
// in the isolated source process. Native compilation keeps its input
// classification and normal object-path rewriting.
func logosCgoSourceName(filename string) string {
	if base.LogosCgoEject {
		filename, _ = objabi.ApplyRewrites(filename, base.Flag.TrimPath)
	}
	return filename
}

// logosConfigureCgoEject enables the checked source form of C imports.
// This mode has a separate compiler process: it cannot replace native
// cgo's pointer checks, object output, or package-lowering state.
func logosConfigureCgoEject(conf *types2.Config, files []*syntax.File) {
	if !base.LogosCgoEject {
		return
	}
	if base.LogosEject == "" {
		base.Errorf("logos.cgoeject requires logos.eject")
		base.ErrorExit()
	}
	hasC := false
	for _, file := range files {
		if !logosfile.Is(file.Pos().FileBase().Filename()) {
			continue
		}
		for _, decl := range file.DeclList {
			if imp, ok := decl.(*syntax.ImportDecl); ok && imp.Path.Value == `"C"` {
				hasC = true
			}
		}
	}
	if !hasC || !conf.LogosUnions {
		base.Errorf("logos.cgoeject requires original loGo/s cgo sources")
		base.ErrorExit()
	}
	types2.LogosEnableCgo(conf)
}

// logosFinishCgoEject runs only after source checking, shared lowering,
// and any required Go recheck have succeeded. C selectors belong in the
// emitted source; they must never reach the native unified-IR writer.
func logosFinishCgoEject(files []*syntax.File) {
	if base.LogosCgoEject {
		writeLogosEject(files)
		base.Exit(0)
	}
}
