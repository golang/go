// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package framepointer defines an Analyzer that reports assembly code
// that clobbers the frame pointer before saving it.
package framepointer

import (
	"go/build"
	"regexp"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysis/analyzerutil"
)

const Doc = "report assembly that clobbers the frame pointer before saving it"

var Analyzer = &analysis.Analyzer{
	Name: "framepointer",
	Doc:  Doc,
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/framepointer",
	Run:  run,
}

// Per-architecture checks for instructions.
// Assume comments, leading and trailing spaces are removed.
type arch struct {
	isFPWrite             func(string) bool
	isFPRead              func(string) bool
	isUnconditionalBranch func(string) bool
}

var re = regexp.MustCompile

func hasAnyPrefix(s string, prefixes ...string) bool {
	for _, p := range prefixes {
		if strings.HasPrefix(s, p) {
			return true
		}
	}
	return false
}

var arches = map[string]arch{
	"amd64": {
		isFPWrite: re(`,\s*BP$`).MatchString, // TODO: can have false positive, e.g. for TESTQ BP,BP. Seems unlikely.
		isFPRead:  re(`\bBP\b`).MatchString,
		isUnconditionalBranch: func(s string) bool {
			return hasAnyPrefix(s, "JMP", "RET")
		},
	},
	"arm64": {
		isFPWrite: func(s string) bool {
			if i := strings.LastIndex(s, ","); i > 0 && strings.HasSuffix(s[i:], "R29") {
				return true
			}
			if hasAnyPrefix(s, "LDP", "LDAXP", "LDXP", "CASP") {
				// Instructions which write to a pair of registers, e.g.
				//	LDP 8(R0), (R26, R29)
				//	CASPD (R2, R3), (R2), (R26, R29)
				lp := strings.LastIndex(s, "(")
				rp := strings.LastIndex(s, ")")
				if lp > -1 && lp < rp {
					return strings.Contains(s[lp:rp], ",") && strings.Contains(s[lp:rp], "R29")
				}
			}
			return false
		},
		isFPRead: re(`\bR29\b`).MatchString,
		isUnconditionalBranch: func(s string) bool {
			// Get just the instruction
			if i := strings.IndexFunc(s, unicode.IsSpace); i > 0 {
				s = s[:i]
			}
			return s == "B" || s == "JMP" || s == "RET"
		},
	},
}

func run(pass *analysis.Pass) (any, error) {
	arch, ok := arches[build.Default.GOARCH]
	if !ok {
		return nil, nil
	}
	if build.Default.GOOS != "linux" && build.Default.GOOS != "darwin" {
		return nil, nil
	}

	// Find assembly files to work on.
	var sfiles []string
	for _, fname := range pass.OtherFiles {
		if strings.HasSuffix(fname, ".s") && pass.Pkg.Path() != "runtime" {
			sfiles = append(sfiles, fname)
		}
	}

	for _, fname := range sfiles {
		content, tf, err := analyzerutil.ReadFile(pass, fname)
		if err != nil {
			return nil, err
		}

		lines := strings.SplitAfter(string(content), "\n")
		active := false
		for lineno, line := range lines {
			lineno++

			// Ignore comments and commented-out code.
			if i := strings.Index(line, "//"); i >= 0 {
				line = line[:i]
			}
			line = strings.TrimSpace(line)
			if line == "" {
				continue
			}

			// We start checking code at a TEXT line for a frameless function.
			if strings.HasPrefix(line, "TEXT") && strings.Contains(line, "(SB)") && strings.Contains(line, "$0") {
				active = true
				continue
			}
			if !active {
				continue
			}

			if arch.isFPWrite(line) {
				pass.Reportf(tf.LineStart(lineno), "frame pointer is clobbered before saving")
				active = false
				continue
			}
			if arch.isFPRead(line) || arch.isUnconditionalBranch(line) {
				active = false
				continue
			}
		}
	}
	return nil, nil
}
