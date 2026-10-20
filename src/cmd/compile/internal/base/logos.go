// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"flag"
)

// LogosEject is the -logos.eject output directory. After checkFiles
// succeeds, the compiler writes one ejected Go file per logos source by
// applying the lowering edits to the original bytes. Empty means no eject.
var LogosEject string

// Logos reports whether the package being compiled came from loGo/s
// sources. It is set during typecheck (noder) and read at archive-write
// time so a plain-Go archive stays byte-identical to upstream: the
// __.LOGOS sidecar exists only when the package used the language.
var Logos bool

// LogosMode selects a compiler helper mode: next rewrites Go to loGo/s,
// context prints one symbol context. Empty compiles normally.
var LogosMode string

// LogosNextOut is the -logos.nextout sidecar path. After checkFiles
// succeeds with -logos.mode=next, the compiler appends one JSON line
// describing the package migration rewrites to this file. Empty means
// no report.
var LogosNextOut string

// LogosCfg is the -logos.cfg file: the logos.toml [lints] policy as JSON,
// written by the go command. A loGo/s package runs its lints on every
// compile when it is set; a Go package runs them only with LogosLintOnly.
// Empty means no lints.
var LogosCfg string

// LogosJSON is the -logos.json flag: every compiler diagnostic prints as
// one JSON object per line (SPEC Sec 16) instead of text.
var LogosJSON bool

// LogosLintOnly is the -logos.lintonly flag, passed only by logos lint:
// run the lints of a Go or loGo/s package, run escape analysis for the
// escape rules, and stop before SSA and the backend.
var LogosLintOnly bool

// logosRegisterFlags registers the loGo/s compiler flags.
func logosRegisterFlags() {
	flag.StringVar(&LogosCfg, "logos.cfg", "", "read the lint policy from `file`")
	flag.BoolVar(&LogosJSON, "logos.json", false, "print diagnostics as JSON lines")
	flag.BoolVar(&LogosLintOnly, "logos.lintonly", false, "lint the package and stop after escape analysis")
	flag.StringVar(&LogosEject, "logos.eject", "", "write ejected Go sources to `directory`")
	flag.StringVar(&LogosMode, "logos.mode", "", "run a logos helper mode (`next` or `context`)")
	flag.StringVar(&LogosNextOut, "logos.nextout", "", "append logos next scan results to `file`")
}
