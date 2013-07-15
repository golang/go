package ssa_test

// This file defines tests of the source and source_ast utilities.

// TODO(adonovan): exhaustive tests that run over the whole input
// tree, not just handcrafted examples.

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"regexp"
	"strings"
	"testing"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/ssa"
)

// -------- Tests of source_ast.go -------------------------------------

// pathToString returns a string containing the concrete types of the
// nodes in path.
func pathToString(path []ast.Node) string {
	var buf bytes.Buffer
	fmt.Fprint(&buf, "[")
	for i, n := range path {
		if i > 0 {
			fmt.Fprint(&buf, " ")
		}
		fmt.Fprint(&buf, strings.TrimPrefix(fmt.Sprintf("%T", n), "*ast."))
	}
	fmt.Fprint(&buf, "]")
	return buf.String()
}

// findInterval parses input and returns the [start, end) positions of
// the first occurrence of substr in input.  f==nil indicates failure;
// an error has already been reported in that case.
//
func findInterval(t *testing.T, fset *token.FileSet, input, substr string) (f *ast.File, start, end token.Pos) {
	f, err := parser.ParseFile(fset, "<input>", input, parser.DeclarationErrors)
	if err != nil {
		t.Errorf("parse error: %s", err)
		return
	}

	i := strings.Index(input, substr)
	if i < 0 {
		t.Errorf("%q is not a substring of input", substr)
		f = nil
		return
	}

	filePos := fset.File(f.Package)
	return f, filePos.Pos(i), filePos.Pos(i + len(substr))
}

// Common input for following tests.
const input = `
// Hello.
package main
import "fmt"
func f() {}
func main() {
	z := (x + y) // add them
        f() // NB: ExprStmt and its CallExpr have same Pos/End
}
`

func TestPathEnclosingInterval_Exact(t *testing.T) {
	// For the exact tests, we check that a substring is mapped to
	// the canonical string for the node it denotes.
	tests := []struct {
		substr string // first occurrence of this string indicates interval
		node   string // complete text of expected containing node
	}{
		{"package",
			input[11 : len(input)-1]},
		{"\npack",
			input[11 : len(input)-1]},
		{"main",
			"main"},
		{"import",
			"import \"fmt\""},
		{"\"fmt\"",
			"\"fmt\""},
		{"\nfunc f() {}\n",
			"func f() {}"},
		{"x ",
			"x"},
		{" y",
			"y"},
		{"z",
			"z"},
		{" + ",
			"x + y"},
		{" :=",
			"z := (x + y)"},
		{"x + y",
			"x + y"},
		{"(x + y)",
			"(x + y)"},
		{" (x + y) ",
			"(x + y)"},
		{" (x + y) // add",
			"(x + y)"},
		{"func",
			"func f() {}"},
		{"func f() {}",
			"func f() {}"},
		{"\nfun",
			"func f() {}"},
		{" f",
			"f"},
	}
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := ssa.PathEnclosingInterval(f, start, end)
		if !exact {
			t.Errorf("PathEnclosingInterval(%q) not exact", test.substr)
			continue
		}

		if len(path) == 0 {
			if test.node != "" {
				t.Errorf("PathEnclosingInterval(%q).path: got [], want %q",
					test.substr, test.node)
			}
			continue
		}

		if got := input[path[0].Pos():path[0].End()]; got != test.node {
			t.Errorf("PathEnclosingInterval(%q): got %q, want %q (path was %s)",
				test.substr, got, test.node, pathToString(path))
			continue
		}
	}
}

func TestPathEnclosingInterval_Paths(t *testing.T) {
	// For these tests, we check only the path of the enclosing
	// node, but not its complete text because it's often quite
	// large when !exact.
	tests := []struct {
		substr string // first occurrence of this string indicates interval
		path   string // the pathToString(),exact of the expected path
	}{
		{"// add",
			"[BlockStmt FuncDecl File],false"},
		{"(x + y",
			"[ParenExpr AssignStmt BlockStmt FuncDecl File],false"},
		{"x +",
			"[BinaryExpr ParenExpr AssignStmt BlockStmt FuncDecl File],false"},
		{"z := (x",
			"[AssignStmt BlockStmt FuncDecl File],false"},
		{"func f",
			"[FuncDecl File],false"},
		{"func f()",
			"[FuncDecl File],false"},
		{" f()",
			"[FuncDecl File],false"},
		{"() {}",
			"[FuncDecl File],false"},
		{"// Hello",
			"[File],false"},
		{" f",
			"[Ident FuncDecl File],true"},
		{"func ",
			"[FuncDecl File],true"},
		{"mai",
			"[Ident File],true"},
		{"f() // NB",
			"[CallExpr ExprStmt BlockStmt FuncDecl File],true"},
	}
	for _, test := range tests {
		f, start, end := findInterval(t, new(token.FileSet), input, test.substr)
		if f == nil {
			continue
		}

		path, exact := ssa.PathEnclosingInterval(f, start, end)
		if got := fmt.Sprintf("%s,%v", pathToString(path), exact); got != test.path {
			t.Errorf("PathEnclosingInterval(%q): got %q, want %q",
				test.substr, got, test.path)
			continue
		}
	}
}

// -------- Tests of source.go -----------------------------------------

func TestEnclosingFunction(t *testing.T) {
	tests := []struct {
		input  string // the input file
		substr string // first occurrence of this string denotes interval
		fn     string // name of expected containing function
	}{
		// We use distinctive numbers as syntactic landmarks.

		// Ordinary function:
		{`package main
		  func f() { println(1003) }`,
			"100", "main.f"},
		// Methods:
		{`package main
                  type T int
		  func (t T) f() { println(200) }`,
			"200", "(main.T).f"},
		// Function literal:
		{`package main
		  func f() { println(func() { print(300) }) }`,
			"300", "func@2.24"},
		// Doubly nested
		{`package main
		  func f() { println(func() { print(func() { print(350) })})}`,
			"350", "func@2.39"},
		// Implicit init for package-level var initializer.
		{"package main; var a = 400", "400", "main.init"},
		// No code for constants:
		{"package main; const a = 500", "500", "(none)"},
		// Explicit init()
		{"package main; func init() { println(600) }", "600", "main.init"},
		// Multiple explicit init functions:
		{`package main
		  func init() { println("foo") }
		  func init() { println(800) }`,
			"800", "main.init"},
		// init() containing FuncLit.
		{`package main
		  func init() { println(func(){print(900)}) }`,
			"900", "func@2.27"},
	}
	for _, test := range tests {
		imp := importer.New(new(importer.Context)) // (NB: no Loader)
		f, start, end := findInterval(t, imp.Fset, test.input, test.substr)
		if f == nil {
			continue
		}
		path, exact := ssa.PathEnclosingInterval(f, start, end)
		if !exact {
			t.Errorf("EnclosingFunction(%q) not exact", test.substr)
			continue
		}
		info, err := imp.CreateSourcePackage("main", []*ast.File{f})
		if err != nil {
			t.Error(err.Error())
			continue
		}

		prog := ssa.NewProgram(imp.Fset, 0)
		prog.CreatePackages(imp)
		pkg := prog.Package(info.Pkg)
		pkg.Build()

		name := "(none)"
		fn := ssa.EnclosingFunction(pkg, path)
		if fn != nil {
			name = fn.String()
		}

		if name != test.fn {
			t.Errorf("EnclosingFunction(%q in %q) got %s, want %s",
				test.substr, test.input, name, test.fn)
			continue
		}

		// While we're here: test HasEnclosingFunction.
		if has := ssa.HasEnclosingFunction(pkg, path); has != (fn != nil) {
			t.Errorf("HasEnclosingFunction(%q in %q) got %v, want %v",
				test.substr, test.input, has, fn != nil)
			continue
		}
	}
}

func TestObjValueLookup(t *testing.T) {
	imp := importer.New(new(importer.Context)) // (uses GCImporter)
	f, err := parser.ParseFile(imp.Fset, "testdata/objlookup.go", nil, parser.DeclarationErrors|parser.ParseComments)
	if err != nil {
		t.Errorf("parse error: %s", err)
		return
	}

	// Maps each var Ident (represented "name:linenum") to the
	// kind of ssa.Value we expect (represented "Literal", "&Alloc").
	expectations := make(map[string]string)

	// Find all annotations of form x::BinOp, &y::Alloc, etc.
	re := regexp.MustCompile(`(\b|&)?(\w*)::(\w*)\b`)
	for _, c := range f.Comments {
		text := c.Text()
		pos := imp.Fset.Position(c.Pos())
		fmt.Println(pos.Line, text)
		for _, m := range re.FindAllStringSubmatch(text, -1) {
			key := fmt.Sprintf("%s:%d", m[2], pos.Line)
			value := m[1] + m[3]
			expectations[key] = value
		}
	}

	info, err := imp.CreateSourcePackage("main", []*ast.File{f})
	if err != nil {
		t.Error(err.Error())
		return
	}

	prog := ssa.NewProgram(imp.Fset, ssa.DebugInfo /*|ssa.LogFunctions*/)
	prog.CreatePackages(imp)
	pkg := prog.Package(info.Pkg)
	pkg.Build()

	// Gather all idents and objects in file.
	objs := make(map[types.Object]bool)
	var ids []*ast.Ident
	ast.Inspect(f, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			ids = append(ids, id)
			if obj := info.ObjectOf(id); obj != nil {
				objs[obj] = true
			}
		}
		return true
	})

	// Check invariants for func and const objects.
	for obj := range objs {
		switch obj := obj.(type) {
		case *types.Func:
			if obj.Name() == "interfaceMethod" {
				continue // TODO(adonovan): not yet implemented.
			}
			checkFuncValue(t, prog, obj)

		case *types.Const:
			checkConstValue(t, prog, obj)
		}
	}

	// Check invariants for var objects.
	// The result varies based on the specific Ident.
	for _, id := range ids {
		if obj, ok := info.ObjectOf(id).(*types.Var); ok {
			ref, _ := ssa.PathEnclosingInterval(f, id.Pos(), id.Pos())
			pos := imp.Fset.Position(id.Pos())
			exp := expectations[fmt.Sprintf("%s:%d", id.Name, pos.Line)]
			if exp == "" {
				t.Errorf("%s: no expectation for var ident %s ", pos, id.Name)
				continue
			}
			wantAddr := false
			if exp[0] == '&' {
				wantAddr = true
				exp = exp[1:]
			}
			checkVarValue(t, prog, ref, obj, exp, wantAddr)
		}
	}
}

func checkFuncValue(t *testing.T, prog *ssa.Program, obj *types.Func) {
	v := prog.FuncValue(obj)
	// fmt.Printf("FuncValue(%s) = %s\n", obj, v) // debugging
	if v == nil {
		t.Errorf("FuncValue(%s) == nil", obj)
		return
	}
	// v must be an *ssa.Function or *ssa.Builtin.
	v2, _ := v.(interface {
		Object() types.Object
	})
	if v2 == nil {
		t.Errorf("FuncValue(%s) = %s %T; has no Object() method",
			obj, v.Name(), v)
		return
	}
	if vobj := v2.Object(); vobj != obj {
		t.Errorf("FuncValue(%s).Object() == %s; value was %s",
			obj, vobj, v.Name())
		return
	}
	if !types.IsIdentical(v.Type(), obj.Type()) {
		t.Errorf("FuncValue(%s).Type() == %s", obj, v.Type())
		return
	}
}

func checkConstValue(t *testing.T, prog *ssa.Program, obj *types.Const) {
	lit := prog.ConstValue(obj)
	// fmt.Printf("ConstValue(%s) = %s\n", obj, lit) // debugging
	if lit == nil {
		t.Errorf("ConstValue(%s) == nil", obj)
		return
	}
	if !types.IsIdentical(lit.Type(), obj.Type()) {
		t.Errorf("ConstValue(%s).Type() == %s", obj, lit.Type())
		return
	}
	if obj.Name() != "nil" {
		if !exact.Compare(lit.Value, token.EQL, obj.Val()) {
			t.Errorf("ConstValue(%s).Value (%s) != %s",
				obj, lit.Value, obj.Val())
			return
		}
	}
}

func checkVarValue(t *testing.T, prog *ssa.Program, ref []ast.Node, obj *types.Var, expKind string, wantAddr bool) {
	// The prefix of all assertions messages.
	prefix := fmt.Sprintf("VarValue(%s @ L%d)",
		obj, prog.Fset.Position(ref[0].Pos()).Line)

	v := prog.VarValue(obj, ref)

	// Kind is the concrete type of the ssa Value.
	gotKind := "nil"
	if v != nil {
		gotKind = fmt.Sprintf("%T", v)[len("*ssa."):]
	}

	// fmt.Printf("%s = %v (kind %q; expect %q) addr=%t\n", prefix, v, gotKind, expKind, wantAddr) // debugging

	// Check the kinds match.
	// "nil" indicates expected failure (e.g. optimized away).
	if expKind != gotKind {
		t.Errorf("%s concrete type == %s, want %s", prefix, gotKind, expKind)
	}

	// Check the types match.
	// If wantAddr, the expected type is the object's address.
	if v != nil {
		expType := obj.Type()
		if wantAddr {
			expType = types.NewPointer(expType)
		}
		if !types.IsIdentical(v.Type(), expType) {
			t.Errorf("%s.Type() == %s, want %s", prefix, v.Type(), expType)
		}
	}
}
