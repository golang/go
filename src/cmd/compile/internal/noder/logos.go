// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"internal/logosfile"
	"path/filepath"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/src"
)

// This file holds the loGo/s source classification for the compiler
// frontend. Upstream files carry only hook lines; all logic lives here.
// The extension set comes from internal/logosfile; the mixed-package
// message matches src/go/build/logos.go exactly.

// isCgoOutput reports whether name is a compiler-generated cgo output.
// Such files are exempt from the mixed-package check.
func isCgoOutput(name string) bool {
	bn := filepath.Base(name)
	return strings.HasPrefix(bn, "_cgo_") || strings.HasSuffix(bn, ".cgo1.go")
}

// isLogosCgoUser reports whether name is the rewritten user output of a
// loGo/s cgo input. cmd/cgo names it from the source file (M20-S0), so
// "c_bridge.lgo" becomes "c_bridge.lgo.cgo1.go". Unlike the other cgo
// outputs it holds the user's own code, written in Go syntax, so the
// package's loGo/s rules apply to it and the compiler lowers it with the
// loGo/s files of the package; "_cgo_*" files stay Go-mode.
func isLogosCgoUser(name string) bool {
	stem, ok := strings.CutSuffix(filepath.Base(name), ".cgo1.go")
	return ok && logosfile.Is(stem)
}

// hasMethodTable reports whether the unified IR reader sees iface as a
// non-empty interface. The reader folds the type set of a closed union
// away (reader.go unionType), so only methods count: a union without
// methods reads as the empty interface, and the itab form that
// types2.Interface.Empty would pick for it leaves the case type of a type
// switch without a runtime type. For every interface of plain Go the two
// tests agree.
func hasMethodTable(iface types2.Type) bool {
	return iface.Underlying().(*types2.Interface).NumMethods() != 0
}

// logosMode returns syntax.LogosMode for loGo/s files and 0 otherwise.
func logosMode(filename string) syntax.Mode {
	if logosfile.Is(filename) {
		return syntax.LogosMode
	}
	return 0
}

// allLogos reports whether the package sources are .lgo files: at least
// one loGo/s file and no user-authored .go files. Cgo outputs are exempt,
// except the rewritten user file of a loGo/s cgo input (M20-S0): its
// //line-mapped source is a .lgo file, so a bridge-only package still
// counts as loGo/s and its platform-pointer stores lower with checks.
func allLogos(noders []*noder) bool {
	sawLogos := false
	for _, p := range noders {
		if p.file == nil {
			continue // parse failed; the upstream error reports it.
		}
		pos := p.file.Pos()
		name := pos.FileBase().Filename()
		if isCgoOutput(name) {
			if logosfile.Is(pos.RelFilename()) {
				sawLogos = true
			}
			continue
		}
		if !logosfile.Is(name) {
			return false
		}
		sawLogos = true
	}
	return sawLogos
}

// checkLogosMixedPackage rejects a package that mixes user-authored .go
// and .lgo files, with the same message as src/go/build/logos.go.
// Cgo outputs are exempt.
func checkLogosMixedPackage(filenames []string) {
	var goFiles, logosFiles []string
	for _, name := range filenames {
		if isCgoOutput(name) {
			continue
		}
		if logosfile.Is(name) {
			logosFiles = append(logosFiles, filepath.Base(name))
		} else if filepath.Ext(name) == ".go" {
			goFiles = append(goFiles, filepath.Base(name))
		}
	}
	if len(goFiles) == 0 || len(logosFiles) == 0 {
		return
	}
	slices.Sort(goFiles)
	slices.Sort(logosFiles)
	base.ErrorfAt(src.NoXPos, 0, "[correctness::mixed_package] cannot mix .go files (%s) and .lgo files (%s) in one package",
		strings.Join(goFiles, ", "), strings.Join(logosFiles, ", "))
	base.ErrorExit()
}
