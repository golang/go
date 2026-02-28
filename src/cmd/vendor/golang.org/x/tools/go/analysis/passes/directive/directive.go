// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package directive defines an Analyzer that checks known Go toolchain directives.
package directive

import (
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/analysis/analyzerutil"
)

const Doc = `check Go toolchain directives such as //go:debug

This analyzer checks for problems with known Go toolchain directives
in all Go source files in a package directory, even those excluded by
//go:build constraints, and all non-Go source files too.

For //go:debug (see https://go.dev/doc/godebug), the analyzer checks
that the directives are placed only in Go source files, only above the
package comment, and only in package main or *_test.go files.

Support for other known directives may be added in the future.

This analyzer does not check //go:build, which is handled by the
buildtag analyzer.
`

var Analyzer = &analysis.Analyzer{
	Name: "directive",
	Doc:  Doc,
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/directive",
	Run:  runDirective,
}

func runDirective(pass *analysis.Pass) (any, error) {
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
				continue
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
	check := newChecker(pass, pass.Fset.File(f.Package).Name(), f)

	for _, group := range f.Comments {
		// A //go:build or a //go:debug comment is ignored after the package declaration
		// (but adjoining it is OK, in contrast to +build comments).
		if group.Pos() >= f.Package {
			check.inHeader = false
		}

		// Check each line of a //-comment.
		for _, c := range group.List {
			check.comment(c.Slash, c.Text)
		}
	}
}

func checkOtherFile(pass *analysis.Pass, filename string) error {
	// We cannot use the Go parser, since is not a Go source file.
	// Read the raw bytes instead.
	content, tf, err := analyzerutil.ReadFile(pass, filename)
	if err != nil {
		return err
	}

	check := newChecker(pass, filename, nil)
	check.nonGoFile(token.Pos(tf.Base()), string(content))
	return nil
}

type checker struct {
	pass     *analysis.Pass
	filename string
	file     *ast.File // nil for non-Go file
	inHeader bool      // in file header (before or adjoining package declaration)
}

func newChecker(pass *analysis.Pass, filename string, file *ast.File) *checker {
	return &checker{
		pass:     pass,
		filename: filename,
		file:     file,
		inHeader: true,
	}
}

func (check *checker) nonGoFile(pos token.Pos, fullText string) {
	// Process each line.
	text := fullText
	inStar := false
	for text != "" {
		offset := len(fullText) - len(text)
		var line string
		line, text, _ = strings.Cut(text, "\n")

		if !inStar && strings.HasPrefix(line, "//") {
			check.comment(pos+token.Pos(offset), line)
			continue
		}

		// Skip over, cut out any /* */ comments,
		// to avoid being confused by a commented-out // comment.
		for {
			line = strings.TrimSpace(line)
			if inStar {
				var ok bool
				_, line, ok = strings.Cut(line, "*/")
				if !ok {
					break
				}
				inStar = false
				continue
			}
			line, inStar = stringsCutPrefix(line, "/*")
			if !inStar {
				break
			}
		}
		if line != "" {
			// Found non-comment non-blank line.
			// Ends space for valid //go:build comments,
			// but also ends the fraction of the file we can
			// reliably parse. From this point on we might
			// incorrectly flag "comments" inside multiline
			// string constants or anything else (this might
			// not even be a Go program). So stop.
			break
		}
	}
}

func (check *checker) comment(pos token.Pos, line string) {
	if !strings.HasPrefix(line, "//go:") {
		return
	}
	// testing hack: stop at // ERROR
	if i := strings.Index(line, " // ERROR "); i >= 0 {
		line = line[:i]
	}

	verb := line
	if i := strings.IndexFunc(verb, unicode.IsSpace); i >= 0 {
		verb = verb[:i]
		if line[i] != ' ' && line[i] != '\t' && line[i] != '\n' {
			r, _ := utf8.DecodeRuneInString(line[i:])
			check.pass.Reportf(pos, "invalid space %#q in %s directive", r, verb)
		}
	}

	switch verb {
	default:
		// TODO: Use the go language version for the file.
		// If that version is not newer than us, then we can
		// report unknown directives.

	case "//go:build":
		// Ignore. The buildtag analyzer reports misplaced comments.

	case "//go:debug":
		if check.file == nil {
			check.pass.Reportf(pos, "//go:debug directive only valid in Go source files")
		} else if check.file.Name.Name != "main" && !strings.HasSuffix(check.filename, "_test.go") {
			check.pass.Reportf(pos, "//go:debug directive only valid in package main or test")
		} else if !check.inHeader {
			check.pass.Reportf(pos, "//go:debug directive only valid before package declaration")
		}
	}
}

// Go 1.20 strings.CutPrefix.
func stringsCutPrefix(s, prefix string) (after string, found bool) {
	if !strings.HasPrefix(s, prefix) {
		return s, false
	}
	return s[len(prefix):], true
}
