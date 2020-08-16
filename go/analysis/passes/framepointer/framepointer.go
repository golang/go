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

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

const Doc = "report assembly that clobbers the frame pointer before saving it"

var Analyzer = &analysis.Analyzer{
	Name: "framepointer",
	Doc:  Doc,
	Run:  run,
}

var (
	re             = regexp.MustCompile
	asmWriteBP     = re(`,\s*BP$`) // TODO: can have false positive, e.g. for TESTQ BP,BP. Seems unlikely.
	asmMentionBP   = re(`\bBP\b`)
	asmControlFlow = re(`^(J|RET)`)
)

func run(pass *analysis.Pass) (interface{}, error) {
	if build.Default.GOARCH != "amd64" { // TODO: arm64 also?
		return nil, nil
	}
	if build.Default.GOOS != "linux" && build.Default.GOOS != "darwin" {
		return nil, nil
	}

	// Find assembly files to work on.
	var sfiles []string
	for _, fname := range pass.OtherFiles {
		if strings.HasSuffix(fname, ".s") {
			sfiles = append(sfiles, fname)
		}
	}

	for _, fname := range sfiles {
		content, tf, err := analysisutil.ReadFile(pass.Fset, fname)
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

			// We start checking code at a TEXT line for a frameless function.
			if strings.HasPrefix(line, "TEXT") && strings.Contains(line, "(SB)") && strings.Contains(line, "$0") {
				active = true
				continue
			}
			if !active {
				continue
			}

			if asmWriteBP.MatchString(line) { // clobber of BP, function is not OK
				pass.Reportf(analysisutil.LineStart(tf, lineno), "frame pointer is clobbered before saving")
				active = false
				continue
			}
			if asmMentionBP.MatchString(line) { // any other use of BP might be a read, so function is OK
				active = false
				continue
			}
			if asmControlFlow.MatchString(line) { // give up after any branch instruction
				active = false
				continue
			}
		}
	}
	return nil, nil
}
