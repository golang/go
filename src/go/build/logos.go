// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package build

import (
	"fmt"
	"internal/logosfile"
	"slices"
	"strings"
)

// This file holds the loGo/s source classification for go/build.
// All substantive loGo/s logic lives here; build.go carries only
// narrow call-site hooks. Every behavior below applies only when the
// Context carries the "logos" tool tag; plain contexts ignore loGo/s
// files exactly as upstream did.
//
// The lineage rule (SPEC Sec 1.1, T-101.1) lives in internal/logosfile:
// the loader recognizes ".lgo" and the three alias extensions after
// stripping trailing U+FE0E/U+FE0F variation selectors. Only the
// comparison sees the stripped form; on-disk names and names recorded
// in Package lists keep their bytes.

// logosMode reports whether the Context enables loGo/s classification.
func (ctxt *Context) logosMode() bool {
	return slices.Contains(ctxt.ToolTags, "logos")
}

// isGoFamilyExt reports whether ext denotes a Go-family source file
// in this Context: ".go" always, loGo/s extensions only in logos mode.
func (ctxt *Context) isGoFamilyExt(ext string) bool {
	if ext == ".go" {
		return true
	}
	return ctxt.logosMode() && logosfile.Is(ext)
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

// logosPackageFiles splits the admitted Go-family files of p into
// plain Go files and loGo/s files, sorted for deterministic errors.
// Only files that reached the buildable lists (GoFiles, CgoFiles,
// TestGoFiles, XTestGoFiles) count: files excluded by build tags sit
// in IgnoredGoFiles and never mix. Callers run this in logos mode only.
func logosPackageFiles(p *Package) (goFiles, logosFiles []string) {
	for _, list := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range list {
			if logosfile.Is(name) {
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
		stem, _, _ := logosfile.Split(name)
		if prev, ok := seen[stem]; ok {
			return fmt.Errorf("[correctness::duplicate_stem_filename] files %s and %s share stem %q",
				prev, name, stem)
		}
		seen[stem] = name
	}
	return nil
}
