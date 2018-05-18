// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package printer

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"io/ioutil"
	"path/filepath"
	"testing"
	"time"
)

const (
	dataDir  = "testdata"
	tabwidth = 8
)

var update = flag.Bool("update", false, "update golden files")

var fset = token.NewFileSet()

type checkMode uint

const (
	export checkMode = 1 << iota
	rawFormat
	idempotent
)

// format parses src, prints the corresponding AST, verifies the resulting
// src is syntactically correct, and returns the resulting src or an error
// if any.
func format(src []byte, mode checkMode) ([]byte, error) {
	// parse src
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		return nil, fmt.Errorf("parse: %s\n%s", err, src)
	}

	// filter exports if necessary
	if mode&export != 0 {
		ast.FileExports(f) // ignore result
		f.Comments = nil   // don't print comments that are not in AST
	}

	// determine printer configuration
	cfg := Config{Tabwidth: tabwidth}
	if mode&rawFormat != 0 {
		cfg.Mode |= RawFormat
	}

	// print AST
	var buf bytes.Buffer
	if err := cfg.Fprint(&buf, fset, f); err != nil {
		return nil, fmt.Errorf("print: %s", err)
	}

	// make sure formatted output is syntactically correct
	res := buf.Bytes()
	if _, err := parser.ParseFile(fset, "", res, 0); err != nil {
		return nil, fmt.Errorf("re-parse: %s\n%s", err, buf.Bytes())
	}

	return res, nil
}

// lineAt returns the line in text starting at offset offs.
func lineAt(text []byte, offs int) []byte {
	i := offs
	for i < len(text) && text[i] != '\n' {
		i++
	}
	return text[offs:i]
}

// diff compares a and b.
func diff(aname, bname string, a, b []byte) error {
	var buf bytes.Buffer // holding long error message

	// compare lengths
	if len(a) != len(b) {
		fmt.Fprintf(&buf, "\nlength changed: len(%s) = %d, len(%s) = %d", aname, len(a), bname, len(b))
	}

	// compare contents
	line := 1
	offs := 1
	for i := 0; i < len(a) && i < len(b); i++ {
		ch := a[i]
		if ch != b[i] {
			fmt.Fprintf(&buf, "\n%s:%d:%d: %s", aname, line, i-offs+1, lineAt(a, offs))
			fmt.Fprintf(&buf, "\n%s:%d:%d: %s", bname, line, i-offs+1, lineAt(b, offs))
			fmt.Fprintf(&buf, "\n\n")
			break
		}
		if ch == '\n' {
			line++
			offs = i + 1
		}
	}

	if buf.Len() > 0 {
		return errors.New(buf.String())
	}
	return nil
}

func runcheck(t *testing.T, source, golden string, mode checkMode) {
	src, err := ioutil.ReadFile(source)
	if err != nil {
		t.Error(err)
		return
	}

	res, err := format(src, mode)
	if err != nil {
		t.Error(err)
		return
	}

	// update golden files if necessary
	if *update {
		if err := ioutil.WriteFile(golden, res, 0644); err != nil {
			t.Error(err)
		}
		return
	}

	// get golden
	gld, err := ioutil.ReadFile(golden)
	if err != nil {
		t.Error(err)
		return
	}

	// formatted source and golden must be the same
	if err := diff(source, golden, res, gld); err != nil {
		t.Error(err)
		return
	}

	if mode&idempotent != 0 {
		// formatting golden must be idempotent
		// (This is very difficult to achieve in general and for now
		// it is only checked for files explicitly marked as such.)
		res, err = format(gld, mode)
		if err := diff(golden, fmt.Sprintf("format(%s)", golden), gld, res); err != nil {
			t.Errorf("golden is not idempotent: %s", err)
		}
	}
}

func check(t *testing.T, source, golden string, mode checkMode) {
	// run the test
	cc := make(chan int)
	go func() {
		runcheck(t, source, golden, mode)
		cc <- 0
	}()

	// wait with timeout
	select {
	case <-time.After(10 * time.Second): // plenty of a safety margin, even for very slow machines
		// test running past time out
		t.Errorf("%s: running too slowly", source)
	case <-cc:
		// test finished within allotted time margin
	}
}

type entry struct {
	source, golden string
	mode           checkMode
}

// Use go test -update to create/update the respective golden files.
var data = []entry{
	{"empty.input", "empty.golden", idempotent},
	{"comments.input", "comments.golden", 0},
	{"comments.input", "comments.x", export},
	{"comments2.input", "comments2.golden", idempotent},
	{"alignment.input", "alignment.golden", idempotent},
	{"linebreaks.input", "linebreaks.golden", idempotent},
	{"expressions.input", "expressions.golden", idempotent},
	{"expressions.input", "expressions.raw", rawFormat | idempotent},
	{"declarations.input", "declarations.golden", 0},
	{"statements.input", "statements.golden", 0},
	{"slow.input", "slow.golden", idempotent},
	{"complit.input", "complit.x", export},
}

func TestFiles(t *testing.T) {
	t.Parallel()
	for _, e := range data {
		source := filepath.Join(dataDir, e.source)
		golden := filepath.Join(dataDir, e.golden)
		mode := e.mode
		t.Run(e.source, func(t *testing.T) {
			t.Parallel()
			check(t, source, golden, mode)
			// TODO(gri) check that golden is idempotent
			//check(t, golden, golden, e.mode)
		})
	}
}

// TestLineComments, using a simple test case, checks that consecutive line
// comments are properly terminated with a newline even if the AST position
// information is incorrect.
//
func TestLineComments(t *testing.T) {
	const src = `// comment 1
	// comment 2
	// comment 3
	package main
	`

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		panic(err) // error in test
	}

	var buf bytes.Buffer
	fset = token.NewFileSet() // use the wrong file set
	Fprint(&buf, fset, f)

	nlines := 0
	for _, ch := range buf.Bytes() {
		if ch == '\n' {
			nlines++
		}
	}

	const expected = 3
	if nlines < expected {
		t.Errorf("got %d, expected %d\n", nlines, expected)
		t.Errorf("result:\n%s", buf.Bytes())
	}
}

// Verify that the printer can be invoked during initialization.
func init() {
	const name = "foobar"
	var buf bytes.Buffer
	if err := Fprint(&buf, fset, &ast.Ident{Name: name}); err != nil {
		panic(err) // error in test
	}
	// in debug mode, the result contains additional information;
	// ignore it
	if s := buf.String(); !debug && s != name {
		panic("got " + s + ", want " + name)
	}
}

// Verify that the printer doesn't crash if the AST contains BadXXX nodes.
func TestBadNodes(t *testing.T) {
	const src = "package p\n("
	const res = "package p\nBadDecl\n"
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err == nil {
		t.Error("expected illegal program") // error in test
	}
	var buf bytes.Buffer
	Fprint(&buf, fset, f)
	if buf.String() != res {
		t.Errorf("got %q, expected %q", buf.String(), res)
	}
}

// testComment verifies that f can be parsed again after printing it
// with its first comment set to comment at any possible source offset.
func testComment(t *testing.T, f *ast.File, srclen int, comment *ast.Comment) {
	f.Comments[0].List[0] = comment
	var buf bytes.Buffer
	for offs := 0; offs <= srclen; offs++ {
		buf.Reset()
		// Printing f should result in a correct program no
		// matter what the (incorrect) comment position is.
		if err := Fprint(&buf, fset, f); err != nil {
			t.Error(err)
		}
		if _, err := parser.ParseFile(fset, "", buf.Bytes(), 0); err != nil {
			t.Fatalf("incorrect program for pos = %d:\n%s", comment.Slash, buf.String())
		}
		// Position information is just an offset.
		// Move comment one byte down in the source.
		comment.Slash++
	}
}

// Verify that the printer produces a correct program
// even if the position information of comments introducing newlines
// is incorrect.
func TestBadComments(t *testing.T) {
	t.Parallel()
	const src = `
// first comment - text and position changed by test
package p
import "fmt"
const pi = 3.14 // rough circle
var (
	x, y, z int = 1, 2, 3
	u, v float64
)
func fibo(n int) {
	if n < 2 {
		return n /* seed values */
	}
	return fibo(n-1) + fibo(n-2)
}
`

	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Error(err) // error in test
	}

	comment := f.Comments[0].List[0]
	pos := comment.Pos()
	if fset.PositionFor(pos, false /* absolute position */).Offset != 1 {
		t.Error("expected offset 1") // error in test
	}

	testComment(t, f, len(src), &ast.Comment{Slash: pos, Text: "//-style comment"})
	testComment(t, f, len(src), &ast.Comment{Slash: pos, Text: "/*-style comment */"})
	testComment(t, f, len(src), &ast.Comment{Slash: pos, Text: "/*-style \n comment */"})
	testComment(t, f, len(src), &ast.Comment{Slash: pos, Text: "/*-style comment \n\n\n */"})
}

type visitor chan *ast.Ident

func (v visitor) Visit(n ast.Node) (w ast.Visitor) {
	if ident, ok := n.(*ast.Ident); ok {
		v <- ident
	}
	return v
}

// idents is an iterator that returns all idents in f via the result channel.
func idents(f *ast.File) <-chan *ast.Ident {
	v := make(visitor)
	go func() {
		ast.Walk(v, f)
		close(v)
	}()
	return v
}

// identCount returns the number of identifiers found in f.
func identCount(f *ast.File) int {
	n := 0
	for range idents(f) {
		n++
	}
	return n
}

// Verify that the SourcePos mode emits correct //line directives
// by testing that position information for matching identifiers
// is maintained.
func TestSourcePos(t *testing.T) {
	const src = `
package p
import ( "go/printer"; "math" )
const pi = 3.14; var x = 0
type t struct{ x, y, z int; u, v, w float32 }
func (t *t) foo(a, b, c int) int {
	return a*t.x + b*t.y +
		// two extra lines here
		// ...
		c*t.z
}
`

	// parse original
	f1, err := parser.ParseFile(fset, "src", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	// pretty-print original
	var buf bytes.Buffer
	err = (&Config{Mode: UseSpaces | SourcePos, Tabwidth: 8}).Fprint(&buf, fset, f1)
	if err != nil {
		t.Fatal(err)
	}

	// parse pretty printed original
	// (//line directives must be interpreted even w/o parser.ParseComments set)
	f2, err := parser.ParseFile(fset, "", buf.Bytes(), 0)
	if err != nil {
		t.Fatalf("%s\n%s", err, buf.Bytes())
	}

	// At this point the position information of identifiers in f2 should
	// match the position information of corresponding identifiers in f1.

	// number of identifiers must be > 0 (test should run) and must match
	n1 := identCount(f1)
	n2 := identCount(f2)
	if n1 == 0 {
		t.Fatal("got no idents")
	}
	if n2 != n1 {
		t.Errorf("got %d idents; want %d", n2, n1)
	}

	// verify that all identifiers have correct line information
	i2range := idents(f2)
	for i1 := range idents(f1) {
		i2 := <-i2range

		if i2.Name != i1.Name {
			t.Errorf("got ident %s; want %s", i2.Name, i1.Name)
		}

		// here we care about the relative (line-directive adjusted) positions
		l1 := fset.Position(i1.Pos()).Line
		l2 := fset.Position(i2.Pos()).Line
		if l2 != l1 {
			t.Errorf("got line %d; want %d for %s", l2, l1, i1.Name)
		}
	}

	if t.Failed() {
		t.Logf("\n%s", buf.Bytes())
	}
}

// Verify that the SourcePos mode doesn't emit unnecessary //line directives
// before empty lines.
func TestIssue5945(t *testing.T) {
	const orig = `
package p   // line 2
func f() {} // line 3

var x, y, z int


func g() { // line 8
}
`

	const want = `//line src.go:2
package p

//line src.go:3
func f() {}

var x, y, z int

//line src.go:8
func g() {
}
`

	// parse original
	f1, err := parser.ParseFile(fset, "src.go", orig, 0)
	if err != nil {
		t.Fatal(err)
	}

	// pretty-print original
	var buf bytes.Buffer
	err = (&Config{Mode: UseSpaces | SourcePos, Tabwidth: 8}).Fprint(&buf, fset, f1)
	if err != nil {
		t.Fatal(err)
	}
	got := buf.String()

	// compare original with desired output
	if got != want {
		t.Errorf("got:\n%s\nwant:\n%s\n", got, want)
	}
}

var decls = []string{
	`import "fmt"`,
	"const pi = 3.1415\nconst e = 2.71828\n\nvar x = pi",
	"func sum(x, y int) int\t{ return x + y }",
}

func TestDeclLists(t *testing.T) {
	for _, src := range decls {
		file, err := parser.ParseFile(fset, "", "package p;"+src, parser.ParseComments)
		if err != nil {
			panic(err) // error in test
		}

		var buf bytes.Buffer
		err = Fprint(&buf, fset, file.Decls) // only print declarations
		if err != nil {
			panic(err) // error in test
		}

		out := buf.String()
		if out != src {
			t.Errorf("\ngot : %q\nwant: %q\n", out, src)
		}
	}
}

var stmts = []string{
	"i := 0",
	"select {}\nvar a, b = 1, 2\nreturn a + b",
	"go f()\ndefer func() {}()",
}

func TestStmtLists(t *testing.T) {
	for _, src := range stmts {
		file, err := parser.ParseFile(fset, "", "package p; func _() {"+src+"}", parser.ParseComments)
		if err != nil {
			panic(err) // error in test
		}

		var buf bytes.Buffer
		err = Fprint(&buf, fset, file.Decls[0].(*ast.FuncDecl).Body.List) // only print statements
		if err != nil {
			panic(err) // error in test
		}

		out := buf.String()
		if out != src {
			t.Errorf("\ngot : %q\nwant: %q\n", out, src)
		}
	}
}

func TestBaseIndent(t *testing.T) {
	t.Parallel()
	// The testfile must not contain multi-line raw strings since those
	// are not indented (because their values must not change) and make
	// this test fail.
	const filename = "printer.go"
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err) // error in test
	}

	file, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		panic(err) // error in test
	}

	for indent := 0; indent < 4; indent++ {
		indent := indent
		t.Run(fmt.Sprint(indent), func(t *testing.T) {
			t.Parallel()
			var buf bytes.Buffer
			(&Config{Tabwidth: tabwidth, Indent: indent}).Fprint(&buf, fset, file)
			// all code must be indented by at least 'indent' tabs
			lines := bytes.Split(buf.Bytes(), []byte{'\n'})
			for i, line := range lines {
				if len(line) == 0 {
					continue // empty lines don't have indentation
				}
				n := 0
				for j, b := range line {
					if b != '\t' {
						// end of indentation
						n = j
						break
					}
				}
				if n < indent {
					t.Errorf("line %d: got only %d tabs; want at least %d: %q", i, n, indent, line)
				}
			}
		})
	}
}

// TestFuncType tests that an ast.FuncType with a nil Params field
// can be printed (per go/ast specification). Test case for issue 3870.
func TestFuncType(t *testing.T) {
	src := &ast.File{
		Name: &ast.Ident{Name: "p"},
		Decls: []ast.Decl{
			&ast.FuncDecl{
				Name: &ast.Ident{Name: "f"},
				Type: &ast.FuncType{},
			},
		},
	}

	var buf bytes.Buffer
	if err := Fprint(&buf, fset, src); err != nil {
		t.Fatal(err)
	}
	got := buf.String()

	const want = `package p

func f()
`

	if got != want {
		t.Fatalf("got:\n%s\nwant:\n%s\n", got, want)
	}
}

type limitWriter struct {
	remaining int
	errCount  int
}

func (l *limitWriter) Write(buf []byte) (n int, err error) {
	n = len(buf)
	if n >= l.remaining {
		n = l.remaining
		err = io.EOF
		l.errCount++
	}
	l.remaining -= n
	return n, err
}

// Test whether the printer stops writing after the first error
func TestWriteErrors(t *testing.T) {
	t.Parallel()
	const filename = "printer.go"
	src, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err) // error in test
	}
	file, err := parser.ParseFile(fset, filename, src, 0)
	if err != nil {
		panic(err) // error in test
	}
	for i := 0; i < 20; i++ {
		lw := &limitWriter{remaining: i}
		err := (&Config{Mode: RawFormat}).Fprint(lw, fset, file)
		if lw.errCount > 1 {
			t.Fatal("Writes continued after first error returned")
		}
		// We expect errCount be 1 iff err is set
		if (lw.errCount != 0) != (err != nil) {
			t.Fatal("Expected err when errCount != 0")
		}
	}
}

// TextX is a skeleton test that can be filled in for debugging one-off cases.
// Do not remove.
func TestX(t *testing.T) {
	const src = `
package p
func _() {}
`
	_, err := format([]byte(src), 0)
	if err != nil {
		t.Error(err)
	}
}

func TestCommentedNode(t *testing.T) {
	const (
		input = `package main

func foo() {
	// comment inside func
}

// leading comment
type bar int // comment2

`

		foo = `func foo() {
	// comment inside func
}`

		bar = `// leading comment
type bar int	// comment2
`
	)

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "input.go", input, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer

	err = Fprint(&buf, fset, &CommentedNode{Node: f.Decls[0], Comments: f.Comments})
	if err != nil {
		t.Fatal(err)
	}

	if buf.String() != foo {
		t.Errorf("got %q, want %q", buf.String(), foo)
	}

	buf.Reset()

	err = Fprint(&buf, fset, &CommentedNode{Node: f.Decls[1], Comments: f.Comments})
	if err != nil {
		t.Fatal(err)
	}

	if buf.String() != bar {
		t.Errorf("got %q, want %q", buf.String(), bar)
	}
}

func TestIssue11151(t *testing.T) {
	const src = "package p\t/*\r/1\r*\r/2*\r\r\r\r/3*\r\r+\r\r/4*/\n"
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	var buf bytes.Buffer
	Fprint(&buf, fset, f)
	got := buf.String()
	const want = "package p\t/*/1*\r/2*\r/3*+/4*/\n" // \r following opening /* should be stripped
	if got != want {
		t.Errorf("\ngot : %q\nwant: %q", got, want)
	}

	// the resulting program must be valid
	_, err = parser.ParseFile(fset, "", got, 0)
	if err != nil {
		t.Errorf("%v\norig: %q\ngot : %q", err, src, got)
	}
}
