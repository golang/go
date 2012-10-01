// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract example functions from file ASTs.

package doc

import (
	"go/ast"
	"go/token"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

type Example struct {
	Name     string // name of the item being exemplified
	Doc      string // example function doc string
	Code     ast.Node
	Play     *ast.File // a whole program version of the example
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
				Play:     playExample(file, f.Body),
				Comments: file.Comments,
				Output:   exampleOutput(f.Body, file.Comments),
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

func exampleOutput(b *ast.BlockStmt, comments []*ast.CommentGroup) string {
	// find the last comment in the function
	var last *ast.CommentGroup
	for _, cg := range comments {
		if cg.Pos() < b.Pos() {
			continue
		}
		if cg.End() > b.End() {
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

// playExample synthesizes a new *ast.File based on the provided
// file with the provided function body as the body of main.
func playExample(file *ast.File, body *ast.BlockStmt) *ast.File {
	if !strings.HasSuffix(file.Name.Name, "_test") {
		// We don't support examples that are part of the
		// greater package (yet).
		return nil
	}

	// Determine the imports we need based on unresolved identifiers.
	// This is a heuristic that presumes package names match base import paths.
	// (Should be good enough most of the time.)
	var unresolved []*ast.Ident
	ast.Inspect(body, func(n ast.Node) bool {
		if e, ok := n.(*ast.SelectorExpr); ok {
			if id, ok := e.X.(*ast.Ident); ok && id.Obj == nil {
				unresolved = append(unresolved, id)
			}
		}
		return true
	})
	imports := make(map[string]string) // [name]path
	for _, s := range file.Imports {
		p, err := strconv.Unquote(s.Path.Value)
		if err != nil {
			continue
		}
		n := path.Base(p)
		if s.Name != nil {
			if s.Name.Name == "." {
				// We can't resolve dot imports (yet).
				return nil
			}
			n = s.Name.Name
		}
		for _, id := range unresolved {
			if n == id.Name {
				imports[n] = p
				break
			}
		}
	}

	// Synthesize new imports.
	importDecl := &ast.GenDecl{
		Tok:    token.IMPORT,
		Lparen: 1, // Need non-zero Lparen and Rparen so that printer
		Rparen: 1, // treats this as a factored import.
	}
	for n, p := range imports {
		s := &ast.ImportSpec{Path: &ast.BasicLit{Value: strconv.Quote(p)}}
		if path.Base(p) != n {
			s.Name = ast.NewIdent(n)
		}
		importDecl.Specs = append(importDecl.Specs, s)
	}

	// TODO(adg): look for other unresolved identifiers and, if found, give up.

	// Filter out comments that are outside the function body.
	var comments []*ast.CommentGroup
	for _, c := range file.Comments {
		if c.Pos() < body.Pos() || c.Pos() >= body.End() {
			continue
		}
		comments = append(comments, c)
	}

	// Strip "Output:" commment and adjust body end position.
	if len(comments) > 0 {
		last := comments[len(comments)-1]
		if outputPrefix.MatchString(last.Text()) {
			comments = comments[:len(comments)-1]
			// Copy body, as the original may be used elsewhere.
			body = &ast.BlockStmt{
				Lbrace: body.Pos(),
				List:   body.List,
				Rbrace: last.Pos(),
			}
		}
	}

	// Synthesize main function.
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent("main"),
		Type: &ast.FuncType{},
		Body: body,
	}

	// Synthesize file.
	f := &ast.File{
		Name:     ast.NewIdent("main"),
		Decls:    []ast.Decl{importDecl, funcDecl},
		Comments: comments,
	}

	// TODO(adg): look for resolved identifiers declared outside function scope
	// and include their declarations in the new file.

	return f
}
