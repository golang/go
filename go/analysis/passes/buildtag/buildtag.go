// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package buildtag

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"
	"io/ioutil"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
)

var Analyzer = &analysis.Analyzer{
	Name: "buildtag",
	Doc:  "check that +build tags are well-formed and correctly located",
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
	content, tf, err := readFile(pass.Fset, filename)
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
			pass.Reportf(lineStart(tf, i+1), "%s", err)
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
	// The original version of this checker in vet could examine
	// files with malformed build tags that would cause the file to
	// be always ignored by "go build". However, drivers for the new
	// analysis API will analyze only the files selected to form a
	// package, so these checks will never fire.
	// TODO(adonovan): rethink this.

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

// -- these declarations are copied from asmdecl --

// readFile reads a file and adds it to the FileSet
// so that we can report errors against it using lineStart.
func readFile(fset *token.FileSet, filename string) ([]byte, *token.File, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, nil, err
	}
	tf := fset.AddFile(filename, -1, len(content))
	tf.SetLinesForContent(content)
	return content, tf, nil
}

// lineStart returns the position of the start of the specified line
// within file f, or NoPos if there is no line of that number.
func lineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.
	//
	// TODO(adonovan): eventually replace this function with the
	// simpler and more efficient (*go/token.File).LineStart, added
	// in go1.12.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}
