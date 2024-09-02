// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package parser

import (
	"fmt"
	"go/ast"
	"go/token"
	"io/fs"
	"strings"
	"testing"
)

var validFiles = []string{
	"parser.go",
	"parser_test.go",
	"error_test.go",
	"short_test.go",
}

func TestParse(t *testing.T) {
	for _, filename := range validFiles {
		_, err := ParseFile(token.NewFileSet(), filename, nil, DeclarationErrors)
		if err != nil {
			t.Fatalf("ParseFile(%s): %v", filename, err)
		}
	}
}

func nameFilter(filename string) bool {
	switch filename {
	case "parser.go", "interface.go", "parser_test.go":
		return true
	case "parser.go.orig":
		return true // permit but should be ignored by ParseDir
	}
	return false
}

func dirFilter(f fs.FileInfo) bool { return nameFilter(f.Name()) }

func TestParseFile(t *testing.T) {
	src := "package p\nvar _=s[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]"
	_, err := ParseFile(token.NewFileSet(), "", src, 0)
	if err == nil {
		t.Errorf("ParseFile(%s) succeeded unexpectedly", src)
	}
}

func TestParseExprFrom(t *testing.T) {
	src := "s[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]+\ns[::]"
	_, err := ParseExprFrom(token.NewFileSet(), "", src, 0)
	if err == nil {
		t.Errorf("ParseExprFrom(%s) succeeded unexpectedly", src)
	}
}

func TestParseDir(t *testing.T) {
	path := "."
	pkgs, err := ParseDir(token.NewFileSet(), path, dirFilter, 0)
	if err != nil {
		t.Fatalf("ParseDir(%s): %v", path, err)
	}
	if n := len(pkgs); n != 1 {
		t.Errorf("got %d packages; want 1", n)
	}
	pkg := pkgs["parser"]
	if pkg == nil {
		t.Errorf(`package "parser" not found`)
		return
	}
	if n := len(pkg.Files); n != 3 {
		t.Errorf("got %d package files; want 3", n)
	}
	for filename := range pkg.Files {
		if !nameFilter(filename) {
			t.Errorf("unexpected package file: %s", filename)
		}
	}
}

func TestIssue42951(t *testing.T) {
	path := "./testdata/issue42951"
	_, err := ParseDir(token.NewFileSet(), path, nil, 0)
	if err != nil {
		t.Errorf("ParseDir(%s): %v", path, err)
	}
}

func TestParseExpr(t *testing.T) {
	// just kicking the tires:
	// a valid arithmetic expression
	src := "a + b"
	x, err := ParseExpr(src)
	if err != nil {
		t.Errorf("ParseExpr(%q): %v", src, err)
	}
	// sanity check
	if _, ok := x.(*ast.BinaryExpr); !ok {
		t.Errorf("ParseExpr(%q): got %T, want *ast.BinaryExpr", src, x)
	}

	// a valid type expression
	src = "struct{x *int}"
	x, err = ParseExpr(src)
	if err != nil {
		t.Errorf("ParseExpr(%q): %v", src, err)
	}
	// sanity check
	if _, ok := x.(*ast.StructType); !ok {
		t.Errorf("ParseExpr(%q): got %T, want *ast.StructType", src, x)
	}

	// an invalid expression
	src = "a + *"
	x, err = ParseExpr(src)
	if err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}
	if x == nil {
		t.Errorf("ParseExpr(%q): got no (partial) result", src)
	}
	if _, ok := x.(*ast.BinaryExpr); !ok {
		t.Errorf("ParseExpr(%q): got %T, want *ast.BinaryExpr", src, x)
	}

	// a valid expression followed by extra tokens is invalid
	src = "a[i] := x"
	if _, err := ParseExpr(src); err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}

	// a semicolon is not permitted unless automatically inserted
	src = "a + b\n"
	if _, err := ParseExpr(src); err != nil {
		t.Errorf("ParseExpr(%q): got error %s", src, err)
	}
	src = "a + b;"
	if _, err := ParseExpr(src); err == nil {
		t.Errorf("ParseExpr(%q): got no error", src)
	}

	// various other stuff following a valid expression
	const validExpr = "a + b"
	const anything = "dh3*#D)#_"
	for _, c := range "!)]};," {
		src := validExpr + string(c) + anything
		if _, err := ParseExpr(src); err == nil {
			t.Errorf("ParseExpr(%q): got no error", src)
		}
	}

	// ParseExpr must not crash
	for _, src := range valids {
		ParseExpr(src)
	}
}

func TestColonEqualsScope(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `package p; func f() { x, y, z := x, y, z }`, 0)
	if err != nil {
		t.Fatal(err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.AssignStmt)
	for _, v := range as.Rhs {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, v := range as.Lhs {
		id := v.(*ast.Ident)
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestVarScope(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `package p; func f() { var x, y, z = x, y, z }`, 0)
	if err != nil {
		t.Fatal(err)
	}

	// RHS refers to undefined globals; LHS does not.
	as := f.Decls[0].(*ast.FuncDecl).Body.List[0].(*ast.DeclStmt).Decl.(*ast.GenDecl).Specs[0].(*ast.ValueSpec)
	for _, v := range as.Values {
		id := v.(*ast.Ident)
		if id.Obj != nil {
			t.Errorf("rhs %s has Obj, should not", id.Name)
		}
	}
	for _, id := range as.Names {
		if id.Obj == nil {
			t.Errorf("lhs %s does not have Obj, should", id.Name)
		}
	}
}

func TestObjects(t *testing.T) {
	const src = `
package p
import fmt "fmt"
const pi = 3.14
type T struct{}
var x int
func f() { L: }
`

	f, err := ParseFile(token.NewFileSet(), "", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	objects := map[string]ast.ObjKind{
		"p":   ast.Bad, // not in a scope
		"fmt": ast.Bad, // not resolved yet
		"pi":  ast.Con,
		"T":   ast.Typ,
		"x":   ast.Var,
		"int": ast.Bad, // not resolved yet
		"f":   ast.Fun,
		"L":   ast.Lbl,
	}

	ast.Inspect(f, func(n ast.Node) bool {
		if ident, ok := n.(*ast.Ident); ok {
			obj := ident.Obj
			if obj == nil {
				if objects[ident.Name] != ast.Bad {
					t.Errorf("no object for %s", ident.Name)
				}
				return true
			}
			if obj.Name != ident.Name {
				t.Errorf("names don't match: obj.Name = %s, ident.Name = %s", obj.Name, ident.Name)
			}
			kind := objects[ident.Name]
			if obj.Kind != kind {
				t.Errorf("%s: obj.Kind = %s; want %s", ident.Name, obj.Kind, kind)
			}
		}
		return true
	})
}

func TestUnresolved(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p
//
func f1a(int)
func f2a(byte, int, float)
func f3a(a, b int, c float)
func f4a(...complex)
func f5a(a s1a, b ...complex)
//
func f1b(*int)
func f2b([]byte, (int), *float)
func f3b(a, b *int, c []float)
func f4b(...*complex)
func f5b(a s1a, b ...[]complex)
//
type s1a struct { int }
type s2a struct { byte; int; s1a }
type s3a struct { a, b int; c float }
//
type s1b struct { *int }
type s2b struct { byte; int; *float }
type s3b struct { a, b *s3b; c []float }
`, 0)
	if err != nil {
		t.Fatal(err)
	}

	want := "int " + // f1a
		"byte int float " + // f2a
		"int float " + // f3a
		"complex " + // f4a
		"complex " + // f5a
		//
		"int " + // f1b
		"byte int float " + // f2b
		"int float " + // f3b
		"complex " + // f4b
		"complex " + // f5b
		//
		"int " + // s1a
		"byte int " + // s2a
		"int float " + // s3a
		//
		"int " + // s1a
		"byte int float " + // s2a
		"float " // s3a

	// collect unresolved identifiers
	var buf strings.Builder
	for _, u := range f.Unresolved {
		buf.WriteString(u.Name)
		buf.WriteByte(' ')
	}
	got := buf.String()

	if got != want {
		t.Errorf("\ngot:  %s\nwant: %s", got, want)
	}
}

func TestCommentGroups(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p /* 1a */ /* 1b */      /* 1c */ // 1d
/* 2a
*/
// 2b
const pi = 3.1415
/* 3a */ // 3b
/* 3c */ const e = 2.7182

// Example from go.dev/issue/3139
func ExampleCount() {
	fmt.Println(strings.Count("cheese", "e"))
	fmt.Println(strings.Count("five", "")) // before & after each rune
	// Output:
	// 3
	// 5
}
`, ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	expected := [][]string{
		{"/* 1a */", "/* 1b */", "/* 1c */", "// 1d"},
		{"/* 2a\n*/", "// 2b"},
		{"/* 3a */", "// 3b", "/* 3c */"},
		{"// Example from go.dev/issue/3139"},
		{"// before & after each rune"},
		{"// Output:", "// 3", "// 5"},
	}
	if len(f.Comments) != len(expected) {
		t.Fatalf("got %d comment groups; expected %d", len(f.Comments), len(expected))
	}
	for i, exp := range expected {
		got := f.Comments[i].List
		if len(got) != len(exp) {
			t.Errorf("got %d comments in group %d; expected %d", len(got), i, len(exp))
			continue
		}
		for j, exp := range exp {
			got := got[j].Text
			if got != exp {
				t.Errorf("got %q in group %d; expected %q", got, i, exp)
			}
		}
	}
}

func getField(file *ast.File, fieldname string) *ast.Field {
	parts := strings.Split(fieldname, ".")
	for _, d := range file.Decls {
		if d, ok := d.(*ast.GenDecl); ok && d.Tok == token.TYPE {
			for _, s := range d.Specs {
				if s, ok := s.(*ast.TypeSpec); ok && s.Name.Name == parts[0] {
					if s, ok := s.Type.(*ast.StructType); ok {
						for _, f := range s.Fields.List {
							for _, name := range f.Names {
								if name.Name == parts[1] {
									return f
								}
							}
						}
					}
				}
			}
		}
	}
	return nil
}

// Don't use ast.CommentGroup.Text() - we want to see exact comment text.
func commentText(c *ast.CommentGroup) string {
	var buf strings.Builder
	if c != nil {
		for _, c := range c.List {
			buf.WriteString(c.Text)
		}
	}
	return buf.String()
}

func checkFieldComments(t *testing.T, file *ast.File, fieldname, lead, line string) {
	f := getField(file, fieldname)
	if f == nil {
		t.Fatalf("field not found: %s", fieldname)
	}
	if got := commentText(f.Doc); got != lead {
		t.Errorf("got lead comment %q; expected %q", got, lead)
	}
	if got := commentText(f.Comment); got != line {
		t.Errorf("got line comment %q; expected %q", got, line)
	}
}

func TestLeadAndLineComments(t *testing.T) {
	f, err := ParseFile(token.NewFileSet(), "", `
package p
type T struct {
	/* F1 lead comment */
	//
	F1 int  /* F1 */ // line comment
	// F2 lead
	// comment
	F2 int  // F2 line comment
	// f3 lead comment
	f3 int  // f3 line comment

	f4 int   /* not a line comment */ ;
        f5 int ; // f5 line comment
	f6 int ; /* f6 line comment */
	f7 int ; /*f7a*/ /*f7b*/ //f7c
}
`, ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	checkFieldComments(t, f, "T.F1", "/* F1 lead comment *///", "/* F1 */// line comment")
	checkFieldComments(t, f, "T.F2", "// F2 lead// comment", "// F2 line comment")
	checkFieldComments(t, f, "T.f3", "// f3 lead comment", "// f3 line comment")
	checkFieldComments(t, f, "T.f4", "", "")
	checkFieldComments(t, f, "T.f5", "", "// f5 line comment")
	checkFieldComments(t, f, "T.f6", "", "/* f6 line comment */")
	checkFieldComments(t, f, "T.f7", "", "/*f7a*//*f7b*///f7c")

	ast.FileExports(f)
	checkFieldComments(t, f, "T.F1", "/* F1 lead comment *///", "/* F1 */// line comment")
	checkFieldComments(t, f, "T.F2", "// F2 lead// comment", "// F2 line comment")
	if getField(f, "T.f3") != nil {
		t.Error("not expected to find T.f3")
	}
}

// TestIssue9979 verifies that empty statements are contained within their enclosing blocks.
func TestIssue9979(t *testing.T) {
	for _, src := range []string{
		"package p; func f() {;}",
		"package p; func f() {L:}",
		"package p; func f() {L:;}",
		"package p; func f() {L:\n}",
		"package p; func f() {L:\n;}",
		"package p; func f() { ; }",
		"package p; func f() { L: }",
		"package p; func f() { L: ; }",
		"package p; func f() { L: \n}",
		"package p; func f() { L: \n; }",
	} {
		fset := token.NewFileSet()
		f, err := ParseFile(fset, "", src, 0)
		if err != nil {
			t.Fatal(err)
		}

		var pos, end token.Pos
		ast.Inspect(f, func(x ast.Node) bool {
			switch s := x.(type) {
			case *ast.BlockStmt:
				pos, end = s.Pos()+1, s.End()-1 // exclude "{", "}"
			case *ast.LabeledStmt:
				pos, end = s.Pos()+2, s.End() // exclude "L:"
			case *ast.EmptyStmt:
				// check containment
				if s.Pos() < pos || s.End() > end {
					t.Errorf("%s: %T[%d, %d] not inside [%d, %d]", src, s, s.Pos(), s.End(), pos, end)
				}
				// check semicolon
				offs := fset.Position(s.Pos()).Offset
				if ch := src[offs]; ch != ';' != s.Implicit {
					want := "want ';'"
					if s.Implicit {
						want = "but ';' is implicit"
					}
					t.Errorf("%s: found %q at offset %d; %s", src, ch, offs, want)
				}
			}
			return true
		})
	}
}

func TestFileStartEndPos(t *testing.T) {
	const src = `// Copyright

//+build tag

// Package p doc comment.
package p

var lastDecl int

/* end of file */
`
	fset := token.NewFileSet()
	f, err := ParseFile(fset, "file.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	// File{Start,End} spans the entire file, not just the declarations.
	if got, want := fset.Position(f.FileStart).String(), "file.go:1:1"; got != want {
		t.Errorf("for File.FileStart, got %s, want %s", got, want)
	}
	// The end position is the newline at the end of the /* end of file */ line.
	if got, want := fset.Position(f.FileEnd).String(), "file.go:10:19"; got != want {
		t.Errorf("for File.FileEnd, got %s, want %s", got, want)
	}
}

// TestIncompleteSelection ensures that an incomplete selector
// expression is parsed as a (blank) *ast.SelectorExpr, not a
// *ast.BadExpr.
func TestIncompleteSelection(t *testing.T) {
	for _, src := range []string{
		"package p; var _ = fmt.",             // at EOF
		"package p; var _ = fmt.\ntype X int", // not at EOF
	} {
		fset := token.NewFileSet()
		f, err := ParseFile(fset, "", src, 0)
		if err == nil {
			t.Errorf("ParseFile(%s) succeeded unexpectedly", src)
			continue
		}

		const wantErr = "expected selector or type assertion"
		if !strings.Contains(err.Error(), wantErr) {
			t.Errorf("ParseFile returned wrong error %q, want %q", err, wantErr)
		}

		var sel *ast.SelectorExpr
		ast.Inspect(f, func(n ast.Node) bool {
			if n, ok := n.(*ast.SelectorExpr); ok {
				sel = n
			}
			return true
		})
		if sel == nil {
			t.Error("found no *ast.SelectorExpr")
			continue
		}
		const wantSel = "&{fmt _}"
		if fmt.Sprint(sel) != wantSel {
			t.Errorf("found selector %s, want %s", sel, wantSel)
			continue
		}
	}
}

func TestLastLineComment(t *testing.T) {
	const src = `package main
type x int // comment
`
	fset := token.NewFileSet()
	f, err := ParseFile(fset, "", src, ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	comment := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.TypeSpec).Comment.List[0].Text
	if comment != "// comment" {
		t.Errorf("got %q, want %q", comment, "// comment")
	}
}

var parseDepthTests = []struct {
	name   string
	format string
	// parseMultiplier is used when a single statement may result in more than one
	// change in the depth level, for instance "1+(..." produces a BinaryExpr
	// followed by a UnaryExpr, which increments the depth twice. The test
	// case comment explains which nodes are triggering the multiple depth
	// changes.
	parseMultiplier int
	// scope is true if we should also test the statement for the resolver scope
	// depth limit.
	scope bool
	// scopeMultiplier does the same as parseMultiplier, but for the scope
	// depths.
	scopeMultiplier int
}{
	// The format expands the part inside « » many times.
	// A second set of brackets nested inside the first stops the repetition,
	// so that for example «(«1»)» expands to (((...((((1))))...))).
	{name: "array", format: "package main; var x «[1]»int"},
	{name: "slice", format: "package main; var x «[]»int"},
	{name: "struct", format: "package main; var x «struct { X «int» }»", scope: true},
	{name: "pointer", format: "package main; var x «*»int"},
	{name: "func", format: "package main; var x «func()»int", scope: true},
	{name: "chan", format: "package main; var x «chan »int"},
	{name: "chan2", format: "package main; var x «<-chan »int"},
	{name: "interface", format: "package main; var x «interface { M() «int» }»", scope: true, scopeMultiplier: 2}, // Scopes: InterfaceType, FuncType
	{name: "map", format: "package main; var x «map[int]»int"},
	{name: "slicelit", format: "package main; var x = «[]any{«»}»", parseMultiplier: 2},             // Parser nodes: UnaryExpr, CompositeLit
	{name: "arraylit", format: "package main; var x = «[1]any{«nil»}»", parseMultiplier: 2},         // Parser nodes: UnaryExpr, CompositeLit
	{name: "structlit", format: "package main; var x = «struct{x any}{«nil»}»", parseMultiplier: 2}, // Parser nodes: UnaryExpr, CompositeLit
	{name: "maplit", format: "package main; var x = «map[int]any{1:«nil»}»", parseMultiplier: 2},    // Parser nodes: CompositeLit, KeyValueExpr
	{name: "dot", format: "package main; var x = «x.»x"},
	{name: "index", format: "package main; var x = x«[1]»"},
	{name: "slice", format: "package main; var x = x«[1:2]»"},
	{name: "slice3", format: "package main; var x = x«[1:2:3]»"},
	{name: "dottype", format: "package main; var x = x«.(any)»"},
	{name: "callseq", format: "package main; var x = x«()»"},
	{name: "methseq", format: "package main; var x = x«.m()»", parseMultiplier: 2}, // Parser nodes: SelectorExpr, CallExpr
	{name: "binary", format: "package main; var x = «1+»1"},
	{name: "binaryparen", format: "package main; var x = «1+(«1»)»", parseMultiplier: 2}, // Parser nodes: BinaryExpr, ParenExpr
	{name: "unary", format: "package main; var x = «^»1"},
	{name: "addr", format: "package main; var x = «& »x"},
	{name: "star", format: "package main; var x = «*»x"},
	{name: "recv", format: "package main; var x = «<-»x"},
	{name: "call", format: "package main; var x = «f(«1»)»", parseMultiplier: 2},    // Parser nodes: Ident, CallExpr
	{name: "conv", format: "package main; var x = «(*T)(«1»)»", parseMultiplier: 2}, // Parser nodes: ParenExpr, CallExpr
	{name: "label", format: "package main; func main() { «Label:» }"},
	{name: "if", format: "package main; func main() { «if true { «» }»}", parseMultiplier: 2, scope: true, scopeMultiplier: 2}, // Parser nodes: IfStmt, BlockStmt. Scopes: IfStmt, BlockStmt
	{name: "ifelse", format: "package main; func main() { «if true {} else » {} }", scope: true},
	{name: "switch", format: "package main; func main() { «switch { default: «» }»}", scope: true, scopeMultiplier: 2},               // Scopes: TypeSwitchStmt, CaseClause
	{name: "typeswitch", format: "package main; func main() { «switch x.(type) { default: «» }» }", scope: true, scopeMultiplier: 2}, // Scopes: TypeSwitchStmt, CaseClause
	{name: "for0", format: "package main; func main() { «for { «» }» }", scope: true, scopeMultiplier: 2},                            // Scopes: ForStmt, BlockStmt
	{name: "for1", format: "package main; func main() { «for x { «» }» }", scope: true, scopeMultiplier: 2},                          // Scopes: ForStmt, BlockStmt
	{name: "for3", format: "package main; func main() { «for f(); g(); h() { «» }» }", scope: true, scopeMultiplier: 2},              // Scopes: ForStmt, BlockStmt
	{name: "forrange0", format: "package main; func main() { «for range x { «» }» }", scope: true, scopeMultiplier: 2},               // Scopes: RangeStmt, BlockStmt
	{name: "forrange1", format: "package main; func main() { «for x = range z { «» }» }", scope: true, scopeMultiplier: 2},           // Scopes: RangeStmt, BlockStmt
	{name: "forrange2", format: "package main; func main() { «for x, y = range z { «» }» }", scope: true, scopeMultiplier: 2},        // Scopes: RangeStmt, BlockStmt
	{name: "go", format: "package main; func main() { «go func() { «» }()» }", parseMultiplier: 2, scope: true},                      // Parser nodes: GoStmt, FuncLit
	{name: "defer", format: "package main; func main() { «defer func() { «» }()» }", parseMultiplier: 2, scope: true},                // Parser nodes: DeferStmt, FuncLit
	{name: "select", format: "package main; func main() { «select { default: «» }» }", scope: true},
}

// split splits pre«mid»post into pre, mid, post.
// If the string does not have that form, split returns x, "", "".
func split(x string) (pre, mid, post string) {
	start, end := strings.Index(x, "«"), strings.LastIndex(x, "»")
	if start < 0 || end < 0 {
		return x, "", ""
	}
	return x[:start], x[start+len("«") : end], x[end+len("»"):]
}

func TestParseDepthLimit(t *testing.T) {
	if testing.Short() {
		t.Skip("test requires significant memory")
	}
	for _, tt := range parseDepthTests {
		for _, size := range []string{"small", "big"} {
			t.Run(tt.name+"/"+size, func(t *testing.T) {
				n := maxNestLev + 1
				if tt.parseMultiplier > 0 {
					n /= tt.parseMultiplier
				}
				if size == "small" {
					// Decrease the number of statements by 10, in order to check
					// that we do not fail when under the limit. 10 is used to
					// provide some wiggle room for cases where the surrounding
					// scaffolding syntax adds some noise to the depth that changes
					// on a per testcase basis.
					n -= 10
				}

				pre, mid, post := split(tt.format)
				if strings.Contains(mid, "«") {
					left, base, right := split(mid)
					mid = strings.Repeat(left, n) + base + strings.Repeat(right, n)
				} else {
					mid = strings.Repeat(mid, n)
				}
				input := pre + mid + post

				fset := token.NewFileSet()
				_, err := ParseFile(fset, "", input, ParseComments|SkipObjectResolution)
				if size == "small" {
					if err != nil {
						t.Errorf("ParseFile(...): %v (want success)", err)
					}
				} else {
					expected := "exceeded max nesting depth"
					if err == nil || !strings.HasSuffix(err.Error(), expected) {
						t.Errorf("ParseFile(...) = _, %v, want %q", err, expected)
					}
				}
			})
		}
	}
}

func TestScopeDepthLimit(t *testing.T) {
	for _, tt := range parseDepthTests {
		if !tt.scope {
			continue
		}
		for _, size := range []string{"small", "big"} {
			t.Run(tt.name+"/"+size, func(t *testing.T) {
				n := maxScopeDepth + 1
				if tt.scopeMultiplier > 0 {
					n /= tt.scopeMultiplier
				}
				if size == "small" {
					// Decrease the number of statements by 10, in order to check
					// that we do not fail when under the limit. 10 is used to
					// provide some wiggle room for cases where the surrounding
					// scaffolding syntax adds some noise to the depth that changes
					// on a per testcase basis.
					n -= 10
				}

				pre, mid, post := split(tt.format)
				if strings.Contains(mid, "«") {
					left, base, right := split(mid)
					mid = strings.Repeat(left, n) + base + strings.Repeat(right, n)
				} else {
					mid = strings.Repeat(mid, n)
				}
				input := pre + mid + post

				fset := token.NewFileSet()
				_, err := ParseFile(fset, "", input, DeclarationErrors)
				if size == "small" {
					if err != nil {
						t.Errorf("ParseFile(...): %v (want success)", err)
					}
				} else {
					expected := "exceeded max scope depth during object resolution"
					if err == nil || !strings.HasSuffix(err.Error(), expected) {
						t.Errorf("ParseFile(...) = _, %v, want %q", err, expected)
					}
				}
			})
		}
	}
}

// proposal go.dev/issue/50429
func TestRangePos(t *testing.T) {
	testcases := []string{
		"package p; func _() { for range x {} }",
		"package p; func _() { for i = range x {} }",
		"package p; func _() { for i := range x {} }",
		"package p; func _() { for k, v = range x {} }",
		"package p; func _() { for k, v := range x {} }",
	}

	for _, src := range testcases {
		fset := token.NewFileSet()
		f, err := ParseFile(fset, src, src, 0)
		if err != nil {
			t.Fatal(err)
		}

		ast.Inspect(f, func(x ast.Node) bool {
			switch s := x.(type) {
			case *ast.RangeStmt:
				pos := fset.Position(s.Range)
				if pos.Offset != strings.Index(src, "range") {
					t.Errorf("%s: got offset %v, want %v", src, pos.Offset, strings.Index(src, "range"))
				}
			}
			return true
		})
	}
}

// TestIssue59180 tests that line number overflow doesn't cause an infinite loop.
func TestIssue59180(t *testing.T) {
	testcases := []string{
		"package p\n//line :9223372036854775806\n\n//",
		"package p\n//line :1:9223372036854775806\n\n//",
		"package p\n//line file:9223372036854775806\n\n//",
	}

	for _, src := range testcases {
		_, err := ParseFile(token.NewFileSet(), "", src, ParseComments)
		if err == nil {
			t.Errorf("ParseFile(%s) succeeded unexpectedly", src)
		}
	}
}

func TestGoVersion(t *testing.T) {
	fset := token.NewFileSet()
	pkgs, err := ParseDir(fset, "./testdata/goversion", nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	for _, p := range pkgs {
		want := strings.ReplaceAll(p.Name, "_", ".")
		if want == "none" {
			want = ""
		}
		for _, f := range p.Files {
			if f.GoVersion != want {
				t.Errorf("%s: GoVersion = %q, want %q", fset.Position(f.Pos()), f.GoVersion, want)
			}
		}
	}
}

func TestIssue57490(t *testing.T) {
	src := `package p; func f() { var x struct` // program not correctly terminated
	fset := token.NewFileSet()
	file, err := ParseFile(fset, "", src, 0)
	if err == nil {
		t.Fatalf("syntax error expected, but no error reported")
	}

	// Because of the syntax error, the end position of the function declaration
	// is past the end of the file's position range.
	funcEnd := file.Decls[0].End()

	// Offset(funcEnd) must not panic (to test panic, set debug=true in token package)
	// (panic: offset 35 out of bounds [0, 34] (position 36 out of bounds [1, 35]))
	tokFile := fset.File(file.Pos())
	offset := tokFile.Offset(funcEnd)
	if offset != tokFile.Size() {
		t.Fatalf("offset = %d, want %d", offset, tokFile.Size())
	}
}

func TestParseTypeParamsAsParenExpr(t *testing.T) {
	const src = "package p\ntype X[A (B),] struct{}"

	fs := token.NewFileSet()
	f, err := ParseFile(fs, "test.go", src, ParseComments|SkipObjectResolution)
	if err != nil {
		t.Fatal(err)
	}
	typeParam := f.Decls[0].(*ast.GenDecl).Specs[0].(*ast.TypeSpec).TypeParams.List[0].Type
	_, ok := typeParam.(*ast.ParenExpr)
	if !ok {
		t.Fatalf("typeParam is a %T; want: *ast.ParenExpr", typeParam)
	}
}
