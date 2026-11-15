// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"fmt"
	"go/build"
	"internal/logosfile"
	"slices"
	"strings"
)

// LoGo/s source classification.
//
// This file mirrors src/go/build/logos.go and shares its extension rule
// through internal/logosfile; modindex keeps no extension list of its
// own. Trailing U+FE0E/U+FE0F variation selectors are ignored when
// matching; on-disk names are never rewritten.
//
// Writers store every recognized kind independently of command mode;
// readers honor logos files only when the requested Context carries the
// "logos" tool tag, so plain-Go reads keep upstream behavior even when
// the shared index stores logos entries (including malformed ones, whose
// per-file errors must not poison ordinary Go imports).

// isLogosTest reports whether name is a loGo/s test file: a recognized
// logos extension whose stem ends in "_test". The "_test" check itself is
// the caller's job, mirroring the HasSuffix "_test.go" rule for ".go".
func isLogosTest(name string) bool {
	stem, _, ok := logosfile.Split(name)
	return ok && strings.HasSuffix(stem, "_test")
}

// logosEnabled reports whether the requested build context carries the
// "logos" tool tag. Readers recognize logos sources only then; plain-Go
// reads ignore stored logos entries entirely, including their errors.
func logosEnabled(ctxt *Context) bool {
	return slices.Contains(ctxt.ToolTags, "logos")
}

// mixedPackageError mirrors the go/build mixed-package diagnostic: every
// admitted file per class, sorted and comma-joined. The single-file case
// reads "cannot mix .go files (a.go) and .lgo files (b.lgo) in one package".
func mixedPackageError(goFiles, logosFiles []string) error {
	return fmt.Errorf("[correctness::mixed_package] cannot mix .go files (%s) and .lgo files (%s) in one package", strings.Join(goFiles, ", "), strings.Join(logosFiles, ", "))
}

// duplicateStemError mirrors the go/build duplicate-stem diagnostic.
func duplicateStemError(first, second, stem string) error {
	return fmt.Errorf("[correctness::duplicate_stem_filename] files %s and %s share stem %q", first, second, stem)
}

// checkLogosPackage mirrors go/build checkLogosPackage over the admitted
// Go-family file lists: mixed classes first, then the first stem collision
// in sorted admitted-logos order. Offenses are recorded through badGoFile
// like any other package error.
func checkLogosPackage(p *build.Package, badGoFile func(string, error)) {
	var goFiles, logosFiles []string
	for _, list := range [][]string{p.GoFiles, p.CgoFiles, p.TestGoFiles, p.XTestGoFiles} {
		for _, name := range list {
			if logosfile.Is(name) {
				logosFiles = append(logosFiles, name)
			} else if strings.HasSuffix(name, ".go") {
				goFiles = append(goFiles, name)
			}
		}
	}
	if len(goFiles) > 0 && len(logosFiles) > 0 {
		slices.Sort(goFiles)
		slices.Sort(logosFiles)
		badGoFile(logosFiles[0], mixedPackageError(goFiles, logosFiles))
		return
	}
	slices.Sort(logosFiles)
	seen := make(map[string]string, len(logosFiles))
	for _, name := range logosFiles {
		stem, _, _ := logosfile.Split(name)
		if prev, ok := seen[stem]; ok {
			badGoFile(name, duplicateStemError(prev, name, stem))
			return
		}
		seen[stem] = name
	}
}
