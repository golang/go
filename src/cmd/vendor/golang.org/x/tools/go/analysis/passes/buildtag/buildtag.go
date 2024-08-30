// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package buildtag defines an Analyzer that checks build tags.
package buildtag

import (
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"strings"
	"unicode"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
)

const Doc = "check //go:build and // +build directives"

var Analyzer = &analysis.Analyzer{
	Name: "buildtag",
	Doc:  Doc,
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/buildtag",
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
			f, err := parser.ParseFile(pass.Fset, name, nil, parser.ParseComments|parser.SkipObjectResolution)
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
	var check checker
	check.init(pass)
	defer check.finish()

	for _, group := range f.Comments {
		// A +build comment is ignored after or adjoining the package declaration.
		if group.End()+1 >= f.Package {
			check.plusBuildOK = false
		}
		// A //go:build comment is ignored after the package declaration
		// (but adjoining it is OK, in contrast to +build comments).
		if group.Pos() >= f.Package {
			check.goBuildOK = false
		}

		// Check each line of a //-comment.
		for _, c := range group.List {
			// "+build" is ignored within or after a /*...*/ comment.
			if !strings.HasPrefix(c.Text, "//") {
				check.plusBuildOK = false
			}
			check.comment(c.Slash, c.Text)
		}
	}
}

func checkOtherFile(pass *analysis.Pass, filename string) error {
	var check checker
	check.init(pass)
	defer check.finish()

	// We cannot use the Go parser, since this may not be a Go source file.
	// Read the raw bytes instead.
	content, tf, err := analysisutil.ReadFile(pass, filename)
	if err != nil {
		return err
	}

	check.file(token.Pos(tf.Base()), string(content))
	return nil
}

type checker struct {
	pass         *analysis.Pass
	plusBuildOK  bool            // "+build" lines still OK
	goBuildOK    bool            // "go:build" lines still OK
	crossCheck   bool            // cross-check go:build and +build lines when done reading file
	inStar       bool            // currently in a /* */ comment
	goBuildPos   token.Pos       // position of first go:build line found
	plusBuildPos token.Pos       // position of first "+build" line found
	goBuild      constraint.Expr // go:build constraint found
	plusBuild    constraint.Expr // AND of +build constraints found
}

func (check *checker) init(pass *analysis.Pass) {
	check.pass = pass
	check.goBuildOK = true
	check.plusBuildOK = true
	check.crossCheck = true
}

func (check *checker) file(pos token.Pos, text string) {
	// Determine cutpoint where +build comments are no longer valid.
	// They are valid in leading // comments in the file followed by
	// a blank line.
	//
	// This must be done as a separate pass because of the
	// requirement that the comment be followed by a blank line.
	var plusBuildCutoff int
	fullText := text
	for text != "" {
		i := strings.Index(text, "\n")
		if i < 0 {
			i = len(text)
		} else {
			i++
		}
		offset := len(fullText) - len(text)
		line := text[:i]
		text = text[i:]
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "//") && line != "" {
			break
		}
		if line == "" {
			plusBuildCutoff = offset
		}
	}

	// Process each line.
	// Must stop once we hit goBuildOK == false
	text = fullText
	check.inStar = false
	for text != "" {
		i := strings.Index(text, "\n")
		if i < 0 {
			i = len(text)
		} else {
			i++
		}
		offset := len(fullText) - len(text)
		line := text[:i]
		text = text[i:]
		check.plusBuildOK = offset < plusBuildCutoff

		if strings.HasPrefix(line, "//") {
			check.comment(pos+token.Pos(offset), line)
			continue
		}

		// Keep looking for the point at which //go:build comments
		// stop being allowed. Skip over, cut out any /* */ comments.
		for {
			line = strings.TrimSpace(line)
			if check.inStar {
				i := strings.Index(line, "*/")
				if i < 0 {
					line = ""
					break
				}
				line = line[i+len("*/"):]
				check.inStar = false
				continue
			}
			if strings.HasPrefix(line, "/*") {
				check.inStar = true
				line = line[len("/*"):]
				continue
			}
			break
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

func (check *checker) comment(pos token.Pos, text string) {
	if strings.HasPrefix(text, "//") {
		if strings.Contains(text, "+build") {
			check.plusBuildLine(pos, text)
		}
		if strings.Contains(text, "//go:build") {
			check.goBuildLine(pos, text)
		}
	}
	if strings.HasPrefix(text, "/*") {
		if i := strings.Index(text, "\n"); i >= 0 {
			// multiline /* */ comment - process interior lines
			check.inStar = true
			i++
			pos += token.Pos(i)
			text = text[i:]
			for text != "" {
				i := strings.Index(text, "\n")
				if i < 0 {
					i = len(text)
				} else {
					i++
				}
				line := text[:i]
				if strings.HasPrefix(line, "//") {
					check.comment(pos, line)
				}
				pos += token.Pos(i)
				text = text[i:]
			}
			check.inStar = false
		}
	}
}

func (check *checker) goBuildLine(pos token.Pos, line string) {
	if !constraint.IsGoBuild(line) {
		if !strings.HasPrefix(line, "//go:build") && constraint.IsGoBuild("//"+strings.TrimSpace(line[len("//"):])) {
			check.pass.Reportf(pos, "malformed //go:build line (space between // and go:build)")
		}
		return
	}
	if !check.goBuildOK || check.inStar {
		check.pass.Reportf(pos, "misplaced //go:build comment")
		check.crossCheck = false
		return
	}

	if check.goBuildPos == token.NoPos {
		check.goBuildPos = pos
	} else {
		check.pass.Reportf(pos, "unexpected extra //go:build line")
		check.crossCheck = false
	}

	// testing hack: stop at // ERROR
	if i := strings.Index(line, " // ERROR "); i >= 0 {
		line = line[:i]
	}

	x, err := constraint.Parse(line)
	if err != nil {
		check.pass.Reportf(pos, "%v", err)
		check.crossCheck = false
		return
	}

	if check.goBuild == nil {
		check.goBuild = x
	}
}

func (check *checker) plusBuildLine(pos token.Pos, line string) {
	line = strings.TrimSpace(line)
	if !constraint.IsPlusBuild(line) {
		// Comment with +build but not at beginning.
		// Only report early in file.
		if check.plusBuildOK && !strings.HasPrefix(line, "// want") {
			check.pass.Reportf(pos, "possible malformed +build comment")
		}
		return
	}
	if !check.plusBuildOK { // inStar implies !plusBuildOK
		check.pass.Reportf(pos, "misplaced +build comment")
		check.crossCheck = false
	}

	if check.plusBuildPos == token.NoPos {
		check.plusBuildPos = pos
	}

	// testing hack: stop at // ERROR
	if i := strings.Index(line, " // ERROR "); i >= 0 {
		line = line[:i]
	}

	fields := strings.Fields(line[len("//"):])
	// IsPlusBuildConstraint check above implies fields[0] == "+build"
	for _, arg := range fields[1:] {
		for _, elem := range strings.Split(arg, ",") {
			if strings.HasPrefix(elem, "!!") {
				check.pass.Reportf(pos, "invalid double negative in build constraint: %s", arg)
				check.crossCheck = false
				continue
			}
			elem = strings.TrimPrefix(elem, "!")
			for _, c := range elem {
				if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
					check.pass.Reportf(pos, "invalid non-alphanumeric build constraint: %s", arg)
					check.crossCheck = false
					break
				}
			}
		}
	}

	if check.crossCheck {
		y, err := constraint.Parse(line)
		if err != nil {
			// Should never happen - constraint.Parse never rejects a // +build line.
			// Also, we just checked the syntax above.
			// Even so, report.
			check.pass.Reportf(pos, "%v", err)
			check.crossCheck = false
			return
		}
		if check.plusBuild == nil {
			check.plusBuild = y
		} else {
			check.plusBuild = &constraint.AndExpr{X: check.plusBuild, Y: y}
		}
	}
}

func (check *checker) finish() {
	if !check.crossCheck || check.plusBuildPos == token.NoPos || check.goBuildPos == token.NoPos {
		return
	}

	// Have both //go:build and // +build,
	// with no errors found (crossCheck still true).
	// Check they match.
	var want constraint.Expr
	lines, err := constraint.PlusBuildLines(check.goBuild)
	if err != nil {
		check.pass.Reportf(check.goBuildPos, "%v", err)
		return
	}
	for _, line := range lines {
		y, err := constraint.Parse(line)
		if err != nil {
			// Definitely should not happen, but not the user's fault.
			// Do not report.
			return
		}
		if want == nil {
			want = y
		} else {
			want = &constraint.AndExpr{X: want, Y: y}
		}
	}
	if want.String() != check.plusBuild.String() {
		check.pass.Reportf(check.plusBuildPos, "+build lines do not match //go:build condition")
		return
	}
}
