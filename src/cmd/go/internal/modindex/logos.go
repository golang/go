// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package modindex

import (
	"fmt"
	"go/build"
	"slices"
	"strings"
)

// LoGo/s source classification.
//
// This file mirrors src/go/build/logos.go: it keeps private copies of the
// extension helpers because modindex mirrors go/build and must not change
// its import structure for the fork. Keep the two files in sync by hand;
// the accepted filename contract lives with the go/build owner.
//
// Recognized extensions are ".lgo" plus the three alias extensions
// U+1F4DC (scroll), U+26A1 (lightning), and U+0295 U+25D4 U+03D6 U+25D4
// U+0294 (gopher), all built below from code points so the file stays
// pure ASCII. Trailing U+FE0E/U+FE0F variation selectors are stripped
// before matching; on-disk names are never rewritten. NFC normalization
// is the identity on all four extension strings, so no normalizer runs.
//
// Writers store every recognized kind independently of command mode;
// readers honor logos files only when the requested Context carries the
// "logos" tool tag, so plain-Go reads keep upstream behavior even when
// the shared index stores logos entries (including malformed ones, whose
// per-file errors must not poison ordinary Go imports).

const logosExtLGO = ".lgo"

// Alias extensions as code points (U+1F4DC scroll, U+26A1 lightning,
// U+0295 U+25D4 U+03D6 U+25D4 U+0294 gopher) so this file stays pure ASCII.
var (
	logosExtScroll = "." + string(rune(0x1F4DC))
	logosExtBolt   = "." + string(rune(0x26A1))
	logosExtGopher = string([]rune{'.', 0x0295, 0x25D4, 0x03D6, 0x25D4, 0x0294})
)

// logosExts lists every recognized loGo/s extension.
var logosExts = []string{logosExtLGO, logosExtScroll, logosExtBolt, logosExtGopher}

// logosVS holds the trailing variation selectors U+FE0E and U+FE0F.
var logosVS = string([]rune{0xFE0E, 0xFE0F})

// stripLogosVS removes trailing Unicode variation selectors U+FE0E and
// U+FE0F so that alias files written with a selector still match.
func stripLogosVS(name string) string {
	return strings.TrimRight(name, logosVS)
}

// logosSplit verifies the stripped name against the recognized extensions
// and returns the stem with the raw (variation-selector-inclusive)
// extension. It reports false when name has no logos extension.
func logosSplit(name string) (stem, ext string, ok bool) {
	trimmed := stripLogosVS(name)
	vs := name[len(trimmed):]
	for _, e := range logosExts {
		if strings.HasSuffix(trimmed, e) {
			return trimmed[:len(trimmed)-len(e)], e + vs, true
		}
	}
	return "", "", false
}

// isLogosExt reports whether name has a loGo/s source extension, after
// stripping trailing variation selectors. It is mode-independent: writers
// store every recognized kind and readers gate on the logos tool tag.
func isLogosExt(name string) bool {
	_, _, ok := logosSplit(name)
	return ok
}

// logosStem mirrors go/build logosStem: the filename minus its full
// on-disk logos extension ("x.<U+26A1><U+FE0F>" becomes "x", keeping any
// "_test" suffix for the caller to check). Non-logos input is returned
// unchanged.
func logosStem(name string) string {
	if stem, _, ok := logosSplit(name); ok {
		return stem
	}
	return name
}

// isLogosTest reports whether name is a loGo/s test file: a recognized
// logos extension whose stem ends in "_test". The "_test" check itself is
// the caller's job, mirroring the HasSuffix "_test.go" rule for ".go".
func isLogosTest(name string) bool {
	stem, _, ok := logosSplit(name)
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
			if isLogosExt(name) {
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
		stem := logosStem(name)
		if prev, ok := seen[stem]; ok {
			badGoFile(name, duplicateStemError(prev, name, stem))
			return
		}
		seen[stem] = name
	}
}
