// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Extract example functions from file ASTs.

package doc

import (
	"cmp"
	"go/ast"
	"go/token"
	"internal/lazyregexp"
	"slices"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

// An Example represents an example function found in a test source file.
type Example struct {
	Name        string // name of the item being exemplified (including optional suffix)
	Suffix      string // example suffix, without leading '_' (only populated by NewFromFiles)
	Doc         string // example function doc string
	Code        ast.Node
	Play        *ast.File // a whole program version of the example
	Comments    []*ast.CommentGroup
	Output      string // expected output
	Unordered   bool
	EmptyOutput bool // expect empty output
	Order       int  // original source code order
}

// Examples returns the examples found in testFiles, sorted by Name field.
// The Order fields record the order in which the examples were encountered.
// The Suffix field is not populated when Examples is called directly, it is
// only populated by [NewFromFiles] for examples it finds in _test.go files.
//
// Playable Examples must be in a package whose name ends in "_test".
// An Example is "playable" (the Play field is non-nil) in either of these
// circumstances:
//   - The example function is self-contained: the function references only
//     identifiers from other packages (or predeclared identifiers, such as
//     "int") and the test file does not include a dot import.
//   - The entire test file is the example: the file contains exactly one
//     example function, zero test, fuzz test, or benchmark function, and at
//     least one top-level function, type, variable, or constant declaration
//     other than the example function.
func Examples(testFiles ...*ast.File) []*Example {
	var list []*Example
	for _, file := range testFiles {
		hasTests := false // file contains tests, fuzz test, or benchmarks
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
			if isTest(name, "Test") || isTest(name, "Benchmark") || isTest(name, "Fuzz") {
				hasTests = true
				continue
			}
			if !isTest(name, "Example") {
				continue
			}
			if params := f.Type.Params; len(params.List) != 0 {
				continue // function has params; not a valid example
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
	slices.SortFunc(list, func(a, b *Example) int {
		return cmp.Compare(a.Name, b.Name)
	})
	return list
}

var outputPrefix = lazyregexp.New(`(?i)^[[:space:]]*(unordered )?output:`)

// Extracts the expected output and whether there was a valid output comment.
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

// isTest tells whether name looks like a test, example, fuzz test, or
// benchmark. It is a Test (say) if there is a character after Test that is not
// a lower-case letter. (We don't want Testiness.)
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
	depDecls, unresolved := findDeclsAndUnresolved(body, topDecls, typMethods)

	// Use unresolved identifiers to determine the imports used by this
	// example. The heuristic assumes package names match base import
	// paths for imports w/o renames (should be good enough most of the time).
	var namedImports []ast.Spec
	var blankImports []ast.Spec // _ imports

	// To preserve the blank lines between groups of imports, find the
	// start position of each group, and assign that position to all
	// imports from that group.
	groupStarts := findImportGroupStarts(file.Imports)
	groupStart := func(s *ast.ImportSpec) token.Pos {
		for i, start := range groupStarts {
			if s.Path.ValuePos < start {
				return groupStarts[i-1]
			}
		}
		return groupStarts[len(groupStarts)-1]
	}

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
		n := assumedPackageName(p)
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
			// Copy the spec and its path to avoid modifying the original.
			spec := *s
			path := *s.Path
			spec.Path = &path
			updateBasicLitPos(spec.Path, groupStart(&spec))
			namedImports = append(namedImports, &spec)
			delete(unresolved, n)
		}
	}

	// Remove predeclared identifiers from unresolved list.
	for n := range unresolved {
		if predeclaredTypes[n] || predeclaredConstants[n] || predeclaredFuncs[n] {
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
	importDecl.Specs = append(namedImports, blankImports...)

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

	slices.SortFunc(decls, func(a, b ast.Decl) int {
		return cmp.Compare(a.Pos(), b.Pos())
	})
	slices.SortFunc(comments, func(a, b *ast.CommentGroup) int {
		return cmp.Compare(a.Pos(), b.Pos())
	})

	// Synthesize file.
	return &ast.File{
		Name:     ast.NewIdent("main"),
		Decls:    decls,
		Comments: comments,
	}
}

// findDeclsAndUnresolved returns all the top-level declarations mentioned in
// the body, and a set of unresolved symbols (those that appear in the body but
// have no declaration in the program).
//
// topDecls maps objects to the top-level declaration declaring them (not
// necessarily obj.Decl, as obj.Decl will be a Spec for GenDecls, but
// topDecls[obj] will be the GenDecl itself).
func findDeclsAndUnresolved(body ast.Node, topDecls map[*ast.Object]ast.Decl, typMethods map[string][]ast.Decl) ([]ast.Decl, map[string]bool) {
	// This function recursively finds every top-level declaration used
	// transitively by the body, populating usedDecls and usedObjs. Then it
	// trims down the declarations to include only the symbols actually
	// referenced by the body.

	unresolved := make(map[string]bool)
	var depDecls []ast.Decl
	usedDecls := make(map[ast.Decl]bool)   // set of top-level decls reachable from the body
	usedObjs := make(map[*ast.Object]bool) // set of objects reachable from the body (each declared by a usedDecl)

	var inspectFunc func(ast.Node) bool
	inspectFunc = func(n ast.Node) bool {
		switch e := n.(type) {
		case *ast.Ident:
			if e.Obj == nil && e.Name != "_" {
				unresolved[e.Name] = true
			} else if d := topDecls[e.Obj]; d != nil {

				usedObjs[e.Obj] = true
				if !usedDecls[d] {
					usedDecls[d] = true
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

	inspectFieldList := func(fl *ast.FieldList) {
		if fl != nil {
			for _, f := range fl.List {
				ast.Inspect(f.Type, inspectFunc)
			}
		}
	}

	// Find the decls immediately referenced by body.
	ast.Inspect(body, inspectFunc)
	// Now loop over them, adding to the list when we find a new decl that the
	// body depends on. Keep going until we don't find anything new.
	for i := 0; i < len(depDecls); i++ {
		switch d := depDecls[i].(type) {
		case *ast.FuncDecl:
			// Inspect type parameters.
			inspectFieldList(d.Type.TypeParams)
			// Inspect types of parameters and results. See #28492.
			inspectFieldList(d.Type.Params)
			inspectFieldList(d.Type.Results)

			// Functions might not have a body. See #42706.
			if d.Body != nil {
				ast.Inspect(d.Body, inspectFunc)
			}
		case *ast.GenDecl:
			for _, spec := range d.Specs {
				switch s := spec.(type) {
				case *ast.TypeSpec:
					inspectFieldList(s.TypeParams)
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

	// Some decls include multiple specs, such as a variable declaration with
	// multiple variables on the same line, or a parenthesized declaration. Trim
	// the declarations to include only the specs that are actually mentioned.
	// However, if there is a constant group with iota, leave it all: later
	// constant declarations in the group may have no value and so cannot stand
	// on their own, and removing any constant from the group could change the
	// values of subsequent ones.
	// See testdata/examples/iota.go for a minimal example.
	var ds []ast.Decl
	for _, d := range depDecls {
		switch d := d.(type) {
		case *ast.FuncDecl:
			ds = append(ds, d)
		case *ast.GenDecl:
			containsIota := false // does any spec have iota?
			// Collect all Specs that were mentioned in the example.
			var specs []ast.Spec
			for _, s := range d.Specs {
				switch s := s.(type) {
				case *ast.TypeSpec:
					if usedObjs[s.Name.Obj] {
						specs = append(specs, s)
					}
				case *ast.ValueSpec:
					if !containsIota {
						containsIota = hasIota(s)
					}
					// A ValueSpec may have multiple names (e.g. "var a, b int").
					// Keep only the names that were mentioned in the example.
					// Exception: the multiple names have a single initializer (which
					// would be a function call with multiple return values). In that
					// case, keep everything.
					if len(s.Names) > 1 && len(s.Values) == 1 {
						specs = append(specs, s)
						continue
					}
					ns := *s
					ns.Names = nil
					ns.Values = nil
					for i, n := range s.Names {
						if usedObjs[n.Obj] {
							ns.Names = append(ns.Names, n)
							if s.Values != nil {
								ns.Values = append(ns.Values, s.Values[i])
							}
						}
					}
					if len(ns.Names) > 0 {
						specs = append(specs, &ns)
					}
				}
			}
			if len(specs) > 0 {
				// Constant with iota? Keep it all.
				if d.Tok == token.CONST && containsIota {
					ds = append(ds, d)
				} else {
					// Synthesize a GenDecl with just the Specs we need.
					nd := *d // copy the GenDecl
					nd.Specs = specs
					if len(specs) == 1 {
						// Remove grouping parens if there is only one spec.
						nd.Lparen = 0
					}
					ds = append(ds, &nd)
				}
			}
		}
	}
	return ds, unresolved
}

func hasIota(s ast.Spec) bool {
	for n := range ast.Preorder(s) {
		// Check that this is the special built-in "iota" identifier, not
		// a user-defined shadow.
		if id, ok := n.(*ast.Ident); ok && id.Name == "iota" && id.Obj == nil {
			return true
		}
	}
	return false
}

// findImportGroupStarts finds the start positions of each sequence of import
// specs that are not separated by a blank line.
func findImportGroupStarts(imps []*ast.ImportSpec) []token.Pos {
	startImps := findImportGroupStarts1(imps)
	groupStarts := make([]token.Pos, len(startImps))
	for i, imp := range startImps {
		groupStarts[i] = imp.Pos()
	}
	return groupStarts
}

// Helper for findImportGroupStarts to ease testing.
func findImportGroupStarts1(origImps []*ast.ImportSpec) []*ast.ImportSpec {
	// Copy to avoid mutation.
	imps := make([]*ast.ImportSpec, len(origImps))
	copy(imps, origImps)
	// Assume the imports are sorted by position.
	slices.SortFunc(imps, func(a, b *ast.ImportSpec) int {
		return cmp.Compare(a.Pos(), b.Pos())
	})
	// Assume gofmt has been applied, so there is a blank line between adjacent imps
	// if and only if they are more than 2 positions apart (newline, tab).
	var groupStarts []*ast.ImportSpec
	prevEnd := token.Pos(-2)
	for _, imp := range imps {
		if imp.Pos()-prevEnd > 2 {
			groupStarts = append(groupStarts, imp)
		}
		prevEnd = imp.End()
		// Account for end-of-line comments.
		if imp.Comment != nil {
			prevEnd = imp.Comment.End()
		}
	}
	return groupStarts
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

// classifyExamples classifies examples and assigns them to the Examples field
// of the relevant Func, Type, or Package that the example is associated with.
//
// The classification process is ambiguous in some cases:
//
//   - ExampleFoo_Bar matches a type named Foo_Bar
//     or a method named Foo.Bar.
//   - ExampleFoo_bar matches a type named Foo_bar
//     or Foo (with a "bar" suffix).
//
// Examples with malformed names are not associated with anything.
func classifyExamples(p *Package, examples []*Example) {
	if len(examples) == 0 {
		return
	}
	// Mapping of names for funcs, types, and methods to the example listing.
	ids := make(map[string]*[]*Example)
	ids[""] = &p.Examples // package-level examples have an empty name
	for _, f := range p.Funcs {
		if !token.IsExported(f.Name) {
			continue
		}
		ids[f.Name] = &f.Examples
	}
	for _, t := range p.Types {
		if !token.IsExported(t.Name) {
			continue
		}
		ids[t.Name] = &t.Examples
		for _, f := range t.Funcs {
			if !token.IsExported(f.Name) {
				continue
			}
			ids[f.Name] = &f.Examples
		}
		for _, m := range t.Methods {
			if !token.IsExported(m.Name) {
				continue
			}
			ids[strings.TrimPrefix(nameWithoutInst(m.Recv), "*")+"_"+m.Name] = &m.Examples
		}
	}

	// Group each example with the associated func, type, or method.
	for _, ex := range examples {
		// Consider all possible split points for the suffix
		// by starting at the end of string (no suffix case),
		// then trying all positions that contain a '_' character.
		//
		// An association is made on the first successful match.
		// Examples with malformed names that match nothing are skipped.
		for i := len(ex.Name); i >= 0; i = strings.LastIndexByte(ex.Name[:i], '_') {
			prefix, suffix, ok := splitExampleName(ex.Name, i)
			if !ok {
				continue
			}
			exs, ok := ids[prefix]
			if !ok {
				continue
			}
			ex.Suffix = suffix
			*exs = append(*exs, ex)
			break
		}
	}

	// Sort list of example according to the user-specified suffix name.
	for _, exs := range ids {
		slices.SortFunc(*exs, func(a, b *Example) int {
			return cmp.Compare(a.Suffix, b.Suffix)
		})
	}
}

// nameWithoutInst returns name if name has no brackets. If name contains
// brackets, then it returns name with all the contents between (and including)
// the outermost left and right bracket removed.
//
// Adapted from debug/gosym/symtab.go:Sym.nameWithoutInst.
func nameWithoutInst(name string) string {
	start := strings.Index(name, "[")
	if start < 0 {
		return name
	}
	end := strings.LastIndex(name, "]")
	if end < 0 {
		// Malformed name, should contain closing bracket too.
		return name
	}
	return name[0:start] + name[end+1:]
}

// splitExampleName attempts to split example name s at index i,
// and reports if that produces a valid split. The suffix may be
// absent. Otherwise, it must start with a lower-case letter and
// be preceded by '_'.
//
// One of i == len(s) or s[i] == '_' must be true.
func splitExampleName(s string, i int) (prefix, suffix string, ok bool) {
	if i == len(s) {
		return s, "", true
	}
	if i == len(s)-1 {
		return "", "", false
	}
	prefix, suffix = s[:i], s[i+1:]
	return prefix, suffix, isExampleSuffix(suffix)
}

func isExampleSuffix(s string) bool {
	r, size := utf8.DecodeRuneInString(s)
	return size > 0 && unicode.IsLower(r)
}

// updateBasicLitPos updates lit.Pos,
// ensuring that lit.End is displaced by the same amount.
// (See https://go.dev/issue/76395.)
func updateBasicLitPos(lit *ast.BasicLit, pos token.Pos) {
	len := lit.End() - lit.Pos()
	lit.ValuePos = pos
	if lit.ValueEnd.IsValid() {
		lit.ValueEnd = pos + len
	}
}
