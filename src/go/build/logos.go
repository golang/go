// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"slices"
	"strings"
)

// This file holds the loGo/s source classification for go/build.
// All substantive loGo/s logic lives here; build.go carries only
// narrow call-site hooks. Every behavior below applies only when the
// Context carries the "logos" tool tag; plain contexts ignore loGo/s
// files exactly as upstream did.
//
// The lineage rule (SPEC Sec 1.1, T-101.1): the loader recognizes
// ".lgo" and the three alias extensions (U+1F4DC scroll, U+26A1
// lightning, U+0295 U+25D4 U+03D6 U+25D4 U+0294 gopher) after
// stripping trailing U+FE0E/U+FE0F variation selectors from the
// extension. No normalizer runs: NFC is the identity on all four
// extension strings.

// logosExts lists the loGo/s source extensions, dot included.
var logosExts = []string{".lgo", ".📜", ".⚡", ".ʕ◔ϖ◔ʔ"}

// logosMode reports whether the Context enables loGo/s classification.
func (ctxt *Context) logosMode() bool {
	return slices.Contains(ctxt.ToolTags, "logos")
}

// stripLogosVarSel trims trailing U+FE0E and U+FE0F runes from an
// extension. Only the comparison sees the stripped form; on-disk
// names and names recorded in Package lists keep their bytes.
func stripLogosVarSel(ext string) string {
	for strings.HasSuffix(ext, "\uFE0E") || strings.HasSuffix(ext, "\uFE0F") {
		ext = ext[:len(ext)-3]
	}
	return ext
}

// isLogosExt reports whether ext is a loGo/s extension, after
// stripping trailing variation selectors. Mode-independent: cache
// writers and readers share it.
func isLogosExt(ext string) bool {
	ext = stripLogosVarSel(ext)
	for _, e := range logosExts {
		if ext == e {
			return true
		}
	}
	return false
}

// isGoFamilyExt reports whether ext denotes a Go-family source file
// in this Context: ".go" always, loGo/s extensions only in logos mode.
func (ctxt *Context) isGoFamilyExt(ext string) bool {
	if ext == ".go" {
		return true
	}
	return ctxt.logosMode() && isLogosExt(ext)
}

// isGoFamilyFile reports whether name is a Go-family source file in
// this Context. Plain contexts answer true only for ".go".
func isGoFamilyFile(name string, ctxt *Context) bool {
	return ctxt.isGoFamilyExt(nameExt(name))
}

// isTestFile reports whether name is a test source file in this
// Context: a Go-family file whose stem (name minus its recognized
// extension) ends in "_test" (for example "p_test.go" or
// "p_test.lgo"). Variation selectors stay in the on-disk name but are
// part of the stripped extension, so "x_test.⚡️" has stem "x_test".
func isTestFile(name string, ctxt *Context) bool {
	ext := nameExt(name)
	if !ctxt.isGoFamilyExt(ext) {
		return false
	}
	return strings.HasSuffix(name[:len(name)-len(ext)], "_test")
}

// logosStem returns name with its loGo/s extension removed, keeping
// everything before it (including any "_test" suffix). The removed
// span is the full on-disk extension, variation selectors included,
// so "x.⚡️" yields stem "x". For names without a loGo/s extension it
// returns the name unchanged.
func logosStem(name string) string {
	ext := nameExt(name)
	if !isLogosExt(ext) {
		return name
	}
	return name[:len(name)-len(ext)]
}

// logosPackageFiles splits the admitted Go-family files of p into
// plain Go files and loGo/s files, sorted for deterministic errors.
// Only files that reached the buildable lists (GoFiles, CgoFiles,
// TestGoFiles, XTestGoFiles) count: files excluded by build tags sit
// in IgnoredGoFiles and never mix. Callers run this in logos mode only.
func logosPackageFiles(p *Package) (goFiles, logosFiles []string) {
	for _, list := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range list {
			if isLogosExt(nameExt(name)) {
				logosFiles = append(logosFiles, name)
			} else {
				goFiles = append(goFiles, name)
			}
		}
	}
	slices.Sort(goFiles)
	slices.Sort(logosFiles)
	return goFiles, logosFiles
}

// checkLogosPackage enforces the loGo/s package rules over the
// admitted files: no mixing of user-authored ".go" and loGo/s
// sources (test files count), and no duplicate stems across loGo/s
// extension variants.
func checkLogosPackage(p *Package) error {
	goFiles, logosFiles := logosPackageFiles(p)
	if err := checkLogosMix(goFiles, logosFiles); err != nil {
		return err
	}
	return checkLogosDuplicateStems(logosFiles)
}

// checkLogosMix fails when the admitted set holds both plain Go and
// loGo/s sources.
func checkLogosMix(goFiles, logosFiles []string) error {
	if len(goFiles) == 0 || len(logosFiles) == 0 {
		return nil
	}
	return fmt.Errorf("[correctness::mixed_package] cannot mix .go files (%s) and .lgo files (%s) in one package",
		strings.Join(goFiles, ", "), strings.Join(logosFiles, ", "))
}

// checkLogosDuplicateStems fails when two admitted loGo/s files share
// a stem under different extensions (for example "x.lgo" beside
// "x.📜"). Input must be sorted; the first collision in that order is
// reported.
func checkLogosDuplicateStems(logosFiles []string) error {
	seen := make(map[string]string, len(logosFiles))
	for _, name := range logosFiles {
		stem := logosStem(name)
		if prev, ok := seen[stem]; ok {
			return fmt.Errorf("[correctness::duplicate_stem_filename] files %s and %s share stem %q",
				prev, name, stem)
		}
		seen[stem] = name
	}
	return nil
}
