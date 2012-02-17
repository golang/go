// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract example functions from file ASTs.

package doc

import (
	"go/ast"
	"go/token"
	"regexp"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Example struct {
	Name     string // name of the item being exemplified
	Doc      string // example function doc string
	Code     ast.Node
	Comments []*ast.CommentGroup
	Output   string // expected output
}

func Examples(files ...*ast.File) []*Example {
	var list []*Example
	for _, file := range files {
		hasTests := false // file contains tests or benchmarks
		numDecl := 0      // number of non-import declarations in the file
		var flist []*Example
		for _, decl := range file.Decls {
			if g, ok := decl.(*ast.GenDecl); ok && g.Tok != token.IMPORT {
				numDecl++
				continue
			}
			f, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			numDecl++
			name := f.Name.Name
			if isTest(name, "Test") || isTest(name, "Benchmark") {
				hasTests = true
				continue
			}
			if !isTest(name, "Example") {
				continue
			}
			var doc string
			if f.Doc != nil {
				doc = f.Doc.Text()
			}
			flist = append(flist, &Example{
				Name:     name[len("Example"):],
				Doc:      doc,
				Code:     f.Body,
				Comments: file.Comments,
				Output:   exampleOutput(f, file.Comments),
			})
		}
		if !hasTests && numDecl > 1 && len(flist) == 1 {
			// If this file only has one example function, some
			// other top-level declarations, and no tests or
			// benchmarks, use the whole file as the example.
			flist[0].Code = file
		}
		list = append(list, flist...)
	}
	sort.Sort(exampleByName(list))
	return list
}

var outputPrefix = regexp.MustCompile(`(?i)^[[:space:]]*output:`)

func exampleOutput(fun *ast.FuncDecl, comments []*ast.CommentGroup) string {
	// find the last comment in the function
	var last *ast.CommentGroup
	for _, cg := range comments {
		if cg.Pos() < fun.Pos() {
			continue
		}
		if cg.End() > fun.End() {
			break
		}
		last = cg
	}
	if last != nil {
		// test that it begins with the correct prefix
		text := last.Text()
		if loc := outputPrefix.FindStringIndex(text); loc != nil {
			return strings.TrimSpace(text[loc[1]:])
		}
	}
	return "" // no suitable comment found
}

// isTest tells whether name looks like a test, example, or benchmark.
// It is a Test (say) if there is a character after Test that is not a
// lower-case letter. (We don't want Testiness.)
func isTest(name, prefix string) bool {
	if !strings.HasPrefix(name, prefix) {
		return false
	}
	if len(name) == len(prefix) { // "Test" is ok
		return true
	}
	rune, _ := utf8.DecodeRuneInString(name[len(prefix):])
	return !unicode.IsLower(rune)
}

type exampleByName []*Example

func (s exampleByName) Len() int           { return len(s) }
func (s exampleByName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s exampleByName) Less(i, j int) bool { return s[i].Name < s[j].Name }
