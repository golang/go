// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract example functions from file ASTs.

package doc

import (
	"go/ast"
	"go/token"
	"internal/lazyregexp"
	"path"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// An Example represents an example function found in a source files.
type Example struct {
	Name        string // name of the item being exemplified
	Doc         string // example function doc string
	Code        ast.Node
	Play        *ast.File // a whole program version of the example
	Comments    []*ast.CommentGroup
	Output      string // expected output
	Unordered   bool
	EmptyOutput bool // expect empty output
	Order       int  // original source code order
}

// Examples returns the examples found in the files, sorted by Name field.
// The Order fields record the order in which the examples were encountered.
//
// Playable Examples must be in a package whose name ends in "_test".
// An Example is "playable" (the Play field is non-nil) in either of these
// circumstances:
//   - The example function is self-contained: the function references only
//     identifiers from other packages (or predeclared identifiers, such as
//     "int") and the test file does not include a dot import.
//   - The entire test file is the example: the file contains exactly one
//     example function, zero test or benchmark functions, and at least one
//     top-level function, type, variable, or constant declaration other
//     than the example function.
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
			if !ok || f.Recv != nil {
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
			if f.Body == nil { // ast.File.Body nil dereference (see issue 28044)
				continue
			}
			var doc string
			if f.Doc != nil {
				doc = f.Doc.Text()
			}
			output, unordered, hasOutput := exampleOutput(f.Body, file.Comments)
			flist = append(flist, &Example{
				Name:        name[len("Example"):],
				Doc:         doc,
				Code:        f.Body,
				Play:        playExample(file, f),
				Comments:    file.Comments,
				Output:      output,
				Unordered:   unordered,
				EmptyOutput: output == "" && hasOutput,
				Order:       len(flist),
			})
		}
		if !hasTests && numDecl > 1 && len(flist) == 1 {
			// If this file only has one example function, some
			// other top-level declarations, and no tests or
			// benchmarks, use the whole file as the example.
			flist[0].Code = file
			flist[0].Play = playExampleFile(file)
		}
		list = append(list, flist...)
	}
	// sort by name
	sort.Slice(list, func(i, j int) bool {
		return list[i].Name < list[j].Name
	})
	return list
}

var outputPrefix = lazyregexp.New(`(?i)^[[:space:]]*(unordered )?output:`)

// Extracts the expected output and whether there was a valid output comment
func exampleOutput(b *ast.BlockStmt, comments []*ast.CommentGroup) (output string, unordered, ok bool) {
	if _, last := lastComment(b, comments); last != nil {
		// test that it begins with the correct prefix
		text := last.Text()
		if loc := outputPrefix.FindStringSubmatchIndex(text); loc != nil {
			if loc[2] != -1 {
				unordered = true
			}
			text = text[loc[1]:]
			// Strip zero or more spaces followed by \n or a single space.
			text = strings.TrimLeft(text, " ")
			if len(text) > 0 && text[0] == '\n' {
				text = text[1:]
			}
			return text, unordered, true
		}
	}
	return "", false, false // no suitable comment found
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

// playExample synthesizes a new *ast.File based on the provided
// file with the provided function body as the body of main.
func playExample(file *ast.File, f *ast.FuncDecl) *ast.File {
	body := f.Body

	if !strings.HasSuffix(file.Name.Name, "_test") {
		// We don't support examples that are part of the
		// greater package (yet).
		return nil
	}

	// Collect top-level declarations in the file.
	topDecls := make(map[*ast.Object]ast.Decl)
	typMethods := make(map[string][]ast.Decl)

	for _, decl := range file.Decls {
		switch d := decl.(type) {
		case *ast.FuncDecl:
			if d.Recv == nil {
				topDecls[d.Name.Obj] = d
			} else {
				if len(d.Recv.List) == 1 {
					t := d.Recv.List[0].Type
					tname, _ := baseTypeName(t)
					typMethods[tname] = append(typMethods[tname], d)
				}
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					topDecls[s.Name.Obj] = d
				case *ast.ValueSpec:
					for _, name := range s.Names {
						topDecls[name.Obj] = d
					}
				}
			}
		}
	}

	// Find unresolved identifiers and uses of top-level declarations.
	unresolved := make(map[string]bool)
	var depDecls []ast.Decl
	hasDepDecls := make(map[ast.Decl]bool)

	var inspectFunc func(ast.Node) bool
	inspectFunc = func(n ast.Node) bool {
		switch e := n.(type) {
		case *ast.Ident:
			if e.Obj == nil && e.Name != "_" {
				unresolved[e.Name] = true
			} else if d := topDecls[e.Obj]; d != nil {
				if !hasDepDecls[d] {
					hasDepDecls[d] = true
					depDecls = append(depDecls, d)
				}
			}
			return true
		case *ast.SelectorExpr:
			// For selector expressions, only inspect the left hand side.
			// (For an expression like fmt.Println, only add "fmt" to the
			// set of unresolved names, not "Println".)
			ast.Inspect(e.X, inspectFunc)
			return false
		case *ast.KeyValueExpr:
			// For key value expressions, only inspect the value
			// as the key should be resolved by the type of the
			// composite literal.
			ast.Inspect(e.Value, inspectFunc)
			return false
		}
		return true
	}
	ast.Inspect(body, inspectFunc)
	for i := 0; i < len(depDecls); i++ {
		switch d := depDecls[i].(type) {
		case *ast.FuncDecl:
			// Inspect types of parameters and results. See #28492.
			if d.Type.Params != nil {
				for _, p := range d.Type.Params.List {
					ast.Inspect(p.Type, inspectFunc)
				}
			}
			if d.Type.Results != nil {
				for _, r := range d.Type.Results.List {
					ast.Inspect(r.Type, inspectFunc)
				}
			}

			ast.Inspect(d.Body, inspectFunc)
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					ast.Inspect(s.Type, inspectFunc)

					depDecls = append(depDecls, typMethods[s.Name.Name]...)
				case *ast.ValueSpec:
					if s.Type != nil {
						ast.Inspect(s.Type, inspectFunc)
					}
					for _, val := range s.Values {
						ast.Inspect(val, inspectFunc)
					}
				}
			}
		}
	}

	// Remove predeclared identifiers from unresolved list.
	for n := range unresolved {
		if predeclaredTypes[n] || predeclaredConstants[n] || predeclaredFuncs[n] {
			delete(unresolved, n)
		}
	}

	// Use unresolved identifiers to determine the imports used by this
	// example. The heuristic assumes package names match base import
	// paths for imports w/o renames (should be good enough most of the time).
	namedImports := make(map[string]string) // [name]path
	var blankImports []ast.Spec             // _ imports
	for _, s := range file.Imports {
		p, err := strconv.Unquote(s.Path.Value)
		if err != nil {
			continue
		}
		if p == "syscall/js" {
			// We don't support examples that import syscall/js,
			// because the package syscall/js is not available in the playground.
			return nil
		}
		n := path.Base(p)
		if s.Name != nil {
			n = s.Name.Name
			switch n {
			case "_":
				blankImports = append(blankImports, s)
				continue
			case ".":
				// We can't resolve dot imports (yet).
				return nil
			}
		}
		if unresolved[n] {
			namedImports[n] = p
			delete(unresolved, n)
		}
	}

	// If there are other unresolved identifiers, give up because this
	// synthesized file is not going to build.
	if len(unresolved) > 0 {
		return nil
	}

	// Include documentation belonging to blank imports.
	var comments []*ast.CommentGroup
	for _, s := range blankImports {
		if c := s.(*ast.ImportSpec).Doc; c != nil {
			comments = append(comments, c)
		}
	}

	// Include comments that are inside the function body.
	for _, c := range file.Comments {
		if body.Pos() <= c.Pos() && c.End() <= body.End() {
			comments = append(comments, c)
		}
	}

	// Strip the "Output:" or "Unordered output:" comment and adjust body
	// end position.
	body, comments = stripOutputComment(body, comments)

	// Include documentation belonging to dependent declarations.
	for _, d := range depDecls {
		switch d := d.(type) {
		case *ast.GenDecl:
			if d.Doc != nil {
				comments = append(comments, d.Doc)
			}
		case *ast.FuncDecl:
			if d.Doc != nil {
				comments = append(comments, d.Doc)
			}
		}
	}

	// Synthesize import declaration.
	importDecl := &ast.GenDecl{
		Tok:    token.IMPORT,
		Lparen: 1, // Need non-zero Lparen and Rparen so that printer
		Rparen: 1, // treats this as a factored import.
	}
	for n, p := range namedImports {
		s := &ast.ImportSpec{Path: &ast.BasicLit{Value: strconv.Quote(p)}}
		if path.Base(p) != n {
			s.Name = ast.NewIdent(n)
		}
		importDecl.Specs = append(importDecl.Specs, s)
	}
	importDecl.Specs = append(importDecl.Specs, blankImports...)

	// Synthesize main function.
	funcDecl := &ast.FuncDecl{
		Name: ast.NewIdent("main"),
		Type: f.Type,
		Body: body,
	}

	decls := make([]ast.Decl, 0, 2+len(depDecls))
	decls = append(decls, importDecl)
	decls = append(decls, depDecls...)
	decls = append(decls, funcDecl)

	sort.Slice(decls, func(i, j int) bool {
		return decls[i].Pos() < decls[j].Pos()
	})

	sort.Slice(comments, func(i, j int) bool {
		return comments[i].Pos() < comments[j].Pos()
	})

	// Synthesize file.
	return &ast.File{
		Name:     ast.NewIdent("main"),
		Decls:    decls,
		Comments: comments,
	}
}

// playExampleFile takes a whole file example and synthesizes a new *ast.File
// such that the example is function main in package main.
func playExampleFile(file *ast.File) *ast.File {
	// Strip copyright comment if present.
	comments := file.Comments
	if len(comments) > 0 && strings.HasPrefix(comments[0].Text(), "Copyright") {
		comments = comments[1:]
	}

	// Copy declaration slice, rewriting the ExampleX function to main.
	var decls []ast.Decl
	for _, d := range file.Decls {
		if f, ok := d.(*ast.FuncDecl); ok && isTest(f.Name.Name, "Example") {
			// Copy the FuncDecl, as it may be used elsewhere.
			newF := *f
			newF.Name = ast.NewIdent("main")
			newF.Body, comments = stripOutputComment(f.Body, comments)
			d = &newF
		}
		decls = append(decls, d)
	}

	// Copy the File, as it may be used elsewhere.
	f := *file
	f.Name = ast.NewIdent("main")
	f.Decls = decls
	f.Comments = comments
	return &f
}

// stripOutputComment finds and removes the "Output:" or "Unordered output:"
// comment from body and comments, and adjusts the body block's end position.
func stripOutputComment(body *ast.BlockStmt, comments []*ast.CommentGroup) (*ast.BlockStmt, []*ast.CommentGroup) {
	// Do nothing if there is no "Output:" or "Unordered output:" comment.
	i, last := lastComment(body, comments)
	if last == nil || !outputPrefix.MatchString(last.Text()) {
		return body, comments
	}

	// Copy body and comments, as the originals may be used elsewhere.
	newBody := &ast.BlockStmt{
		Lbrace: body.Lbrace,
		List:   body.List,
		Rbrace: last.Pos(),
	}
	newComments := make([]*ast.CommentGroup, len(comments)-1)
	copy(newComments, comments[:i])
	copy(newComments[i:], comments[i+1:])
	return newBody, newComments
}

// lastComment returns the last comment inside the provided block.
func lastComment(b *ast.BlockStmt, c []*ast.CommentGroup) (i int, last *ast.CommentGroup) {
	if b == nil {
		return
	}
	pos, end := b.Pos(), b.End()
	for j, cg := range c {
		if cg.Pos() < pos {
			continue
		}
		if cg.End() > end {
			break
		}
		i, last = j, cg
	}
	return
}
