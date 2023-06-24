// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Delete this file once Go 1.17 comes out and we can retire Go 1.15 support.

//go:build !go1.16
// +build !go1.16

// Package buildtag defines an Analyzer that checks build tags.
package buildtag

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

const Doc = "check // +build directives"

var Analyzer = &analysis.Analyzer{
	Name: "buildtag",
	Doc:  Doc,
	Run:  runBuildTag,
}

func runBuildTag(pass *analysis.Pass) (interface{}, error) {
	for _, f := range pass.Files {
		checkGoFile(pass, f)
	}
	for _, name := range pass.OtherFiles {
		if err := checkOtherFile(pass, name); err != nil {
			return nil, err
		}
	}
	for _, name := range pass.IgnoredFiles {
		if strings.HasSuffix(name, ".go") {
			f, err := parser.ParseFile(pass.Fset, name, nil, parser.ParseComments)
			if err != nil {
				// Not valid Go source code - not our job to diagnose, so ignore.
				return nil, nil
			}
			checkGoFile(pass, f)
		} else {
			if err := checkOtherFile(pass, name); err != nil {
				return nil, err
			}
		}
	}
	return nil, nil
}

func checkGoFile(pass *analysis.Pass, f *ast.File) {
	pastCutoff := false
	for _, group := range f.Comments {
		// A +build comment is ignored after or adjoining the package declaration.
		if group.End()+1 >= f.Package {
			pastCutoff = true
		}

		// "+build" is ignored within or after a /*...*/ comment.
		if !strings.HasPrefix(group.List[0].Text, "//") {
			pastCutoff = true
			continue
		}

		// Check each line of a //-comment.
		for _, c := range group.List {
			if !strings.Contains(c.Text, "+build") {
				continue
			}
			if err := checkLine(c.Text, pastCutoff); err != nil {
				pass.Reportf(c.Pos(), "%s", err)
			}
		}
	}
}

func checkOtherFile(pass *analysis.Pass, filename string) error {
	content, tf, err := analysisutil.ReadFile(pass.Fset, filename)
	if err != nil {
		return err
	}

	// We must look at the raw lines, as build tags may appear in non-Go
	// files such as assembly files.
	lines := bytes.SplitAfter(content, nl)

	// Determine cutpoint where +build comments are no longer valid.
	// They are valid in leading // comments in the file followed by
	// a blank line.
	//
	// This must be done as a separate pass because of the
	// requirement that the comment be followed by a blank line.
	var cutoff int
	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashSlash) {
			if len(line) > 0 {
				break
			}
			cutoff = i
		}
	}

	for i, line := range lines {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, slashSlash) {
			continue
		}
		if !bytes.Contains(line, []byte("+build")) {
			continue
		}
		if err := checkLine(string(line), i >= cutoff); err != nil {
			pass.Reportf(analysisutil.LineStart(tf, i+1), "%s", err)
			continue
		}
	}
	return nil
}

// checkLine checks a line that starts with "//" and contains "+build".
func checkLine(line string, pastCutoff bool) error {
	line = strings.TrimPrefix(line, "//")
	line = strings.TrimSpace(line)

	if strings.HasPrefix(line, "+build") {
		fields := strings.Fields(line)
		if fields[0] != "+build" {
			// Comment is something like +buildasdf not +build.
			return fmt.Errorf("possible malformed +build comment")
		}
		if pastCutoff {
			return fmt.Errorf("+build comment must appear before package clause and be followed by a blank line")
		}
		if err := checkArguments(fields); err != nil {
			return err
		}
	} else {
		// Comment with +build but not at beginning.
		if !pastCutoff {
			return fmt.Errorf("possible malformed +build comment")
		}
	}
	return nil
}

func checkArguments(fields []string) error {
	for _, arg := range fields[1:] {
		for _, elem := range strings.Split(arg, ",") {
			if strings.HasPrefix(elem, "!!") {
				return fmt.Errorf("invalid double negative in build constraint: %s", arg)
			}
			elem = strings.TrimPrefix(elem, "!")
			for _, c := range elem {
				if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
					return fmt.Errorf("invalid non-alphanumeric build constraint: %s", arg)
				}
			}
		}
	}
	return nil
}

var (
	nl         = []byte("\n")
	slashSlash = []byte("//")
)
