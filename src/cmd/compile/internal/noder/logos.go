// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"path/filepath"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/internal/src"
)

// This file holds the loGo/s source classification for the compiler
// frontend. Upstream files carry only hook lines; all logic lives here.
// The extension set and the mixed-package message match
// src/go/build/logos.go exactly.

// logosExts lists the loGo/s source extensions, dot included.
var logosExts = []string{".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

// stripLogosVarSel trims trailing U+FE0E and U+FE0F runes from an extension.
func stripLogosVarSel(ext string) string {
	for strings.HasSuffix(ext, "\uFE0E") || strings.HasSuffix(ext, "\uFE0F") {
		ext = ext[:len(ext)-3]
	}
	return ext
}

// isLogosFile reports whether name carries a loGo/s extension, after
// stripping trailing variation selectors from the extension.
func isLogosFile(name string) bool {
	ext := stripLogosVarSel(filepath.Ext(name))
	return slices.Contains(logosExts, ext)
}

// isCgoOutput reports whether name is a compiler-generated cgo output.
// Such files are exempt from the mixed-package check.
func isCgoOutput(name string) bool {
	bn := filepath.Base(name)
	return strings.HasPrefix(bn, "_cgo_") || strings.HasSuffix(bn, ".cgo1.go")
}

// logosMode returns syntax.LogosMode for loGo/s files and 0 otherwise.
func logosMode(filename string) syntax.Mode {
	if isLogosFile(filename) {
		return syntax.LogosMode
	}
	return 0
}

// allLogos reports whether the package sources are .lgo files: at least
// one loGo/s file and no user-authored .go files. Cgo outputs are exempt.
func allLogos(noders []*noder) bool {
	sawLogos := false
	for _, p := range noders {
		name := p.file.Pos().FileBase().Filename()
		if isCgoOutput(name) {
			continue
		}
		if !isLogosFile(name) {
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
		if isLogosFile(name) {
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
