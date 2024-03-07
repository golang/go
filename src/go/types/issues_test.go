// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements tests for various issues.

package types_test

import (
	"fmt"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"internal/testenv"
	"regexp"
	"sort"
	"strings"
	"testing"

	. "go/types"
)

func TestIssue5770(t *testing.T) {
	_, err := typecheck(`package p; type S struct{T}`, nil, nil)
	const want = "undefined: T"
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("got: %v; want: %s", err, want)
	}
}

func TestIssue5849(t *testing.T) {
	src := `
package p
var (
	s uint
	_ = uint8(8)
	_ = uint16(16) << s
	_ = uint32(32 << s)
	_ = uint64(64 << s + s)
	_ = (interface{})("foo")
	_ = (interface{})(nil)
)`
	types := make(map[ast.Expr]TypeAndValue)
	mustTypecheck(src, nil, &Info{Types: types})

	for x, tv := range types {
		var want Type
		switch x := x.(type) {
		case *ast.BasicLit:
			switch x.Value {
			case `8`:
				want = Typ[Uint8]
			case `16`:
				want = Typ[Uint16]
			case `32`:
				want = Typ[Uint32]
			case `64`:
				want = Typ[Uint] // because of "+ s", s is of type uint
			case `"foo"`:
				want = Typ[String]
			}
		case *ast.Ident:
			if x.Name == "nil" {
				want = Typ[UntypedNil]
			}
		}
		if want != nil && !Identical(tv.Type, want) {
			t.Errorf("got %s; want %s", tv.Type, want)
		}
	}
}

func TestIssue6413(t *testing.T) {
	src := `
package p
func f() int {
	defer f()
	go f()
	return 0
}
`
	types := make(map[ast.Expr]TypeAndValue)
	mustTypecheck(src, nil, &Info{Types: types})

	want := Typ[Int]
	n := 0
	for x, tv := range types {
		if _, ok := x.(*ast.CallExpr); ok {
			if tv.Type != want {
				t.Errorf("%s: got %s; want %s", fset.Position(x.Pos()), tv.Type, want)
			}
			n++
		}
	}

	if n != 2 {
		t.Errorf("got %d CallExprs; want 2", n)
	}
}

func TestIssue7245(t *testing.T) {
	src := `
package p
func (T) m() (res bool) { return }
type T struct{} // receiver type after method declaration
`
	f := mustParse(fset, src)

	var conf Config
	defs := make(map[*ast.Ident]Object)
	_, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Defs: defs})
	if err != nil {
		t.Fatal(err)
	}

	m := f.Decls[0].(*ast.FuncDecl)
	res1 := defs[m.Name].(*Func).Type().(*Signature).Results().At(0)
	res2 := defs[m.Type.Results.List[0].Names[0]].(*Var)

	if res1 != res2 {
		t.Errorf("got %s (%p) != %s (%p)", res1, res2, res1, res2)
	}
}

// This tests that uses of existing vars on the LHS of an assignment
// are Uses, not Defs; and also that the (illegal) use of a non-var on
// the LHS of an assignment is a Use nonetheless.
func TestIssue7827(t *testing.T) {
	const src = `
package p
func _() {
	const w = 1        // defs w
        x, y := 2, 3       // defs x, y
        w, x, z := 4, 5, 6 // uses w, x, defs z; error: cannot assign to w
        _, _, _ = x, y, z  // uses x, y, z
}
`
	// We need a specific fileset in this test below for positions.
	// Cannot use typecheck helper.
	fset := token.NewFileSet()
	f := mustParse(fset, src)

	const want = `L3 defs func p._()
L4 defs const w untyped int
L5 defs var x int
L5 defs var y int
L6 defs var z int
L6 uses const w untyped int
L6 uses var x int
L7 uses var x int
L7 uses var y int
L7 uses var z int`

	// don't abort at the first error
	conf := Config{Error: func(err error) { t.Log(err) }}
	defs := make(map[*ast.Ident]Object)
	uses := make(map[*ast.Ident]Object)
	_, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, &Info{Defs: defs, Uses: uses})
	if s := err.Error(); !strings.HasSuffix(s, "cannot assign to w") {
		t.Errorf("Check: unexpected error: %s", s)
	}

	var facts []string
	for id, obj := range defs {
		if obj != nil {
			fact := fmt.Sprintf("L%d defs %s", fset.Position(id.Pos()).Line, obj)
			facts = append(facts, fact)
		}
	}
	for id, obj := range uses {
		fact := fmt.Sprintf("L%d uses %s", fset.Position(id.Pos()).Line, obj)
		facts = append(facts, fact)
	}
	sort.Strings(facts)

	got := strings.Join(facts, "\n")
	if got != want {
		t.Errorf("Unexpected defs/uses\ngot:\n%s\nwant:\n%s", got, want)
	}
}

// This tests that the package associated with the types.Object.Pkg method
// is the type's package independent of the order in which the imports are
// listed in the sources src1, src2 below.
// The actual issue is in go/internal/gcimporter which has a corresponding
// test; we leave this test here to verify correct behavior at the go/types
// level.
func TestIssue13898(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	const src0 = `
package main

import "go/types"

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	// like src0, but also imports go/importer
	const src1 = `
package main

import (
	"go/types"
	_ "go/importer"
)

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	// like src1 but with different import order
	// (used to fail with this issue)
	const src2 = `
package main

import (
	_ "go/importer"
	"go/types"
)

func main() {
	var info types.Info
	for _, obj := range info.Uses {
		_ = obj.Pkg()
	}
}
`
	f := func(test, src string) {
		info := &Info{Uses: make(map[*ast.Ident]Object)}
		mustTypecheck(src, nil, info)

		var pkg *Package
		count := 0
		for id, obj := range info.Uses {
			if id.Name == "Pkg" {
				pkg = obj.Pkg()
				count++
			}
		}
		if count != 1 {
			t.Fatalf("%s: got %d entries named Pkg; want 1", test, count)
		}
		if pkg.Name() != "types" {
			t.Fatalf("%s: got %v; want package types", test, pkg)
		}
	}

	f("src0", src0)
	f("src1", src1)
	f("src2", src2)
}

func TestIssue22525(t *testing.T) {
	const src = `package p; func f() { var a, b, c, d, e int }`

	got := "\n"
	conf := Config{Error: func(err error) { got += err.Error() + "\n" }}
	typecheck(src, &conf, nil) // do not crash
	want := `
p:1:27: a declared and not used
p:1:30: b declared and not used
p:1:33: c declared and not used
p:1:36: d declared and not used
p:1:39: e declared and not used
`
	if got != want {
		t.Errorf("got: %swant: %s", got, want)
	}
}

func TestIssue25627(t *testing.T) {
	const prefix = `package p; import "unsafe"; type P *struct{}; type I interface{}; type T `
	// The src strings (without prefix) are constructed such that the number of semicolons
	// plus one corresponds to the number of fields expected in the respective struct.
	for _, src := range []string{
		`struct { x Missing }`,
		`struct { Missing }`,
		`struct { *Missing }`,
		`struct { unsafe.Pointer }`,
		`struct { P }`,
		`struct { *I }`,
		`struct { a int; b Missing; *Missing }`,
	} {
		f := mustParse(fset, prefix+src)

		cfg := Config{Importer: importer.Default(), Error: func(err error) {}}
		info := &Info{Types: make(map[ast.Expr]TypeAndValue)}
		_, err := cfg.Check(f.Name.Name, fset, []*ast.File{f}, info)
		if err != nil {
			if _, ok := err.(Error); !ok {
				t.Fatal(err)
			}
		}

		ast.Inspect(f, func(n ast.Node) bool {
			if spec, _ := n.(*ast.TypeSpec); spec != nil {
				if tv, ok := info.Types[spec.Type]; ok && spec.Name.Name == "T" {
					want := strings.Count(src, ";") + 1
					if got := tv.Type.(*Struct).NumFields(); got != want {
						t.Errorf("%s: got %d fields; want %d", src, got, want)
					}
				}
			}
			return true
		})
	}
}

func TestIssue28005(t *testing.T) {
	// method names must match defining interface name for this test
	// (see last comment in this function)
	sources := [...]string{
		"package p; type A interface{ A() }",
		"package p; type B interface{ B() }",
		"package p; type X interface{ A; B }",
	}

	// compute original file ASTs
	var orig [len(sources)]*ast.File
	for i, src := range sources {
		orig[i] = mustParse(fset, src)
	}

	// run the test for all order permutations of the incoming files
	for _, perm := range [][len(sources)]int{
		{0, 1, 2},
		{0, 2, 1},
		{1, 0, 2},
		{1, 2, 0},
		{2, 0, 1},
		{2, 1, 0},
	} {
		// create file order permutation
		files := make([]*ast.File, len(sources))
		for i := range perm {
			files[i] = orig[perm[i]]
		}

		// type-check package with given file order permutation
		var conf Config
		info := &Info{Defs: make(map[*ast.Ident]Object)}
		_, err := conf.Check("", fset, files, info)
		if err != nil {
			t.Fatal(err)
		}

		// look for interface object X
		var obj Object
		for name, def := range info.Defs {
			if name.Name == "X" {
				obj = def
				break
			}
		}
		if obj == nil {
			t.Fatal("object X not found")
		}
		iface := obj.Type().Underlying().(*Interface) // object X must be an interface

		// Each iface method m is embedded; and m's receiver base type name
		// must match the method's name per the choice in the source file.
		for i := 0; i < iface.NumMethods(); i++ {
			m := iface.Method(i)
			recvName := m.Type().(*Signature).Recv().Type().(*Named).Obj().Name()
			if recvName != m.Name() {
				t.Errorf("perm %v: got recv %s; want %s", perm, recvName, m.Name())
			}
		}
	}
}

func TestIssue28282(t *testing.T) {
	// create type interface { error }
	et := Universe.Lookup("error").Type()
	it := NewInterfaceType(nil, []Type{et})
	it.Complete()
	// verify that after completing the interface, the embedded method remains unchanged
	want := et.Underlying().(*Interface).Method(0)
	got := it.Method(0)
	if got != want {
		t.Fatalf("%s.Method(0): got %q (%p); want %q (%p)", it, got, got, want, want)
	}
	// verify that lookup finds the same method in both interfaces (redundant check)
	obj, _, _ := LookupFieldOrMethod(et, false, nil, "Error")
	if obj != want {
		t.Fatalf("%s.Lookup: got %q (%p); want %q (%p)", et, obj, obj, want, want)
	}
	obj, _, _ = LookupFieldOrMethod(it, false, nil, "Error")
	if obj != want {
		t.Fatalf("%s.Lookup: got %q (%p); want %q (%p)", it, obj, obj, want, want)
	}
}

func TestIssue29029(t *testing.T) {
	f1 := mustParse(fset, `package p; type A interface { M() }`)
	f2 := mustParse(fset, `package p; var B interface { A }`)

	// printInfo prints the *Func definitions recorded in info, one *Func per line.
	printInfo := func(info *Info) string {
		var buf strings.Builder
		for _, obj := range info.Defs {
			if fn, ok := obj.(*Func); ok {
				fmt.Fprintln(&buf, fn)
			}
		}
		return buf.String()
	}

	// The *Func (method) definitions for package p must be the same
	// independent on whether f1 and f2 are type-checked together, or
	// incrementally.

	// type-check together
	var conf Config
	info := &Info{Defs: make(map[*ast.Ident]Object)}
	check := NewChecker(&conf, fset, NewPackage("", "p"), info)
	if err := check.Files([]*ast.File{f1, f2}); err != nil {
		t.Fatal(err)
	}
	want := printInfo(info)

	// type-check incrementally
	info = &Info{Defs: make(map[*ast.Ident]Object)}
	check = NewChecker(&conf, fset, NewPackage("", "p"), info)
	if err := check.Files([]*ast.File{f1}); err != nil {
		t.Fatal(err)
	}
	if err := check.Files([]*ast.File{f2}); err != nil {
		t.Fatal(err)
	}
	got := printInfo(info)

	if got != want {
		t.Errorf("\ngot : %swant: %s", got, want)
	}
}

func TestIssue34151(t *testing.T) {
	const asrc = `package a; type I interface{ M() }; type T struct { F interface { I } }`
	const bsrc = `package b; import "a"; type T struct { F interface { a.I } }; var _ = a.T(T{})`

	a := mustTypecheck(asrc, nil, nil)

	conf := Config{Importer: importHelper{pkg: a}}
	mustTypecheck(bsrc, &conf, nil)
}

type importHelper struct {
	pkg      *Package
	fallback Importer
}

func (h importHelper) Import(path string) (*Package, error) {
	if path == h.pkg.Path() {
		return h.pkg, nil
	}
	if h.fallback == nil {
		return nil, fmt.Errorf("got package path %q; want %q", path, h.pkg.Path())
	}
	return h.fallback.Import(path)
}

// TestIssue34921 verifies that we don't update an imported type's underlying
// type when resolving an underlying type. Specifically, when determining the
// underlying type of b.T (which is the underlying type of a.T, which is int)
// we must not set the underlying type of a.T again since that would lead to
// a race condition if package b is imported elsewhere, in a package that is
// concurrently type-checked.
func TestIssue34921(t *testing.T) {
	defer func() {
		if r := recover(); r != nil {
			t.Error(r)
		}
	}()

	var sources = []string{
		`package a; type T int`,
		`package b; import "a"; type T a.T`,
	}

	var pkg *Package
	for _, src := range sources {
		conf := Config{Importer: importHelper{pkg: pkg}}
		pkg = mustTypecheck(src, &conf, nil) // pkg imported by the next package in this test
	}
}

func TestIssue43088(t *testing.T) {
	// type T1 struct {
	//         _ T2
	// }
	//
	// type T2 struct {
	//         _ struct {
	//                 _ T2
	//         }
	// }
	n1 := NewTypeName(nopos, nil, "T1", nil)
	T1 := NewNamed(n1, nil, nil)
	n2 := NewTypeName(nopos, nil, "T2", nil)
	T2 := NewNamed(n2, nil, nil)
	s1 := NewStruct([]*Var{NewField(nopos, nil, "_", T2, false)}, nil)
	T1.SetUnderlying(s1)
	s2 := NewStruct([]*Var{NewField(nopos, nil, "_", T2, false)}, nil)
	s3 := NewStruct([]*Var{NewField(nopos, nil, "_", s2, false)}, nil)
	T2.SetUnderlying(s3)

	// These calls must terminate (no endless recursion).
	Comparable(T1)
	Comparable(T2)
}

func TestIssue44515(t *testing.T) {
	typ := Unsafe.Scope().Lookup("Pointer").Type()

	got := TypeString(typ, nil)
	want := "unsafe.Pointer"
	if got != want {
		t.Errorf("got %q; want %q", got, want)
	}

	qf := func(pkg *Package) string {
		if pkg == Unsafe {
			return "foo"
		}
		return ""
	}
	got = TypeString(typ, qf)
	want = "foo.Pointer"
	if got != want {
		t.Errorf("got %q; want %q", got, want)
	}
}

func TestIssue43124(t *testing.T) {
	// TODO(rFindley) move this to testdata by enhancing support for importing.

	testenv.MustHaveGoBuild(t) // The go command is needed for the importer to determine the locations of stdlib .a files.

	// All involved packages have the same name (template). Error messages should
	// disambiguate between text/template and html/template by printing the full
	// path.
	const (
		asrc = `package a; import "text/template"; func F(template.Template) {}; func G(int) {}`
		bsrc = `
package b

import (
	"a"
	"html/template"
)

func _() {
	// Packages should be fully qualified when there is ambiguity within the
	// error string itself.
	a.F(template /* ERRORx "cannot use.*html/template.* as .*text/template" */ .Template{})
}
`
		csrc = `
package c

import (
	"a"
	"fmt"
	"html/template"
)

// go.dev/issue/46905: make sure template is not the first package qualified.
var _ fmt.Stringer = 1 // ERRORx "cannot use 1.*as fmt\\.Stringer"

// Packages should be fully qualified when there is ambiguity in reachable
// packages. In this case both a (and for that matter html/template) import
// text/template.
func _() { a.G(template /* ERRORx "cannot use .*html/template.*Template" */ .Template{}) }
`

		tsrc = `
package template

import "text/template"

type T int

// Verify that the current package name also causes disambiguation.
var _ T = template /* ERRORx "cannot use.*text/template.* as T value" */.Template{}
`
	)

	a := mustTypecheck(asrc, nil, nil)
	imp := importHelper{pkg: a, fallback: importer.Default()}

	withImporter := func(cfg *Config) {
		cfg.Importer = imp
	}

	testFiles(t, []string{"b.go"}, [][]byte{[]byte(bsrc)}, false, withImporter)
	testFiles(t, []string{"c.go"}, [][]byte{[]byte(csrc)}, false, withImporter)
	testFiles(t, []string{"t.go"}, [][]byte{[]byte(tsrc)}, false, withImporter)
}

func TestIssue50646(t *testing.T) {
	anyType := Universe.Lookup("any").Type()
	comparableType := Universe.Lookup("comparable").Type()

	if !Comparable(anyType) {
		t.Error("any is not a comparable type")
	}
	if !Comparable(comparableType) {
		t.Error("comparable is not a comparable type")
	}

	if Implements(anyType, comparableType.Underlying().(*Interface)) {
		t.Error("any implements comparable")
	}
	if !Implements(comparableType, anyType.(*Interface)) {
		t.Error("comparable does not implement any")
	}

	if AssignableTo(anyType, comparableType) {
		t.Error("any assignable to comparable")
	}
	if !AssignableTo(comparableType, anyType) {
		t.Error("comparable not assignable to any")
	}
}

func TestIssue55030(t *testing.T) {
	// makeSig makes the signature func(typ...)
	makeSig := func(typ Type) {
		par := NewVar(nopos, nil, "", typ)
		params := NewTuple(par)
		NewSignatureType(nil, nil, nil, params, nil, true)
	}

	// makeSig must not panic for the following (example) types:
	// []int
	makeSig(NewSlice(Typ[Int]))

	// string
	makeSig(Typ[String])

	// P where P's core type is string
	{
		P := NewTypeName(nopos, nil, "P", nil) // [P string]
		makeSig(NewTypeParam(P, NewInterfaceType(nil, []Type{Typ[String]})))
	}

	// P where P's core type is an (unnamed) slice
	{
		P := NewTypeName(nopos, nil, "P", nil) // [P []int]
		makeSig(NewTypeParam(P, NewInterfaceType(nil, []Type{NewSlice(Typ[Int])})))
	}

	// P where P's core type is bytestring (i.e., string or []byte)
	{
		t1 := NewTerm(true, Typ[String])          // ~string
		t2 := NewTerm(false, NewSlice(Typ[Byte])) // []byte
		u := NewUnion([]*Term{t1, t2})            // ~string | []byte
		P := NewTypeName(nopos, nil, "P", nil)    // [P ~string | []byte]
		makeSig(NewTypeParam(P, NewInterfaceType(nil, []Type{u})))
	}
}

func TestIssue51093(t *testing.T) {
	// Each test stands for a conversion of the form P(val)
	// where P is a type parameter with typ as constraint.
	// The test ensures that P(val) has the correct type P
	// and is not a constant.
	var tests = []struct {
		typ string
		val string
	}{
		{"bool", "false"},
		{"int", "-1"},
		{"uint", "1.0"},
		{"rune", "'a'"},
		{"float64", "3.5"},
		{"complex64", "1.25"},
		{"string", "\"foo\""},

		// some more complex constraints
		{"~byte", "1"},
		{"~int | ~float64 | complex128", "1"},
		{"~uint64 | ~rune", "'X'"},
	}

	for _, test := range tests {
		src := fmt.Sprintf("package p; func _[P %s]() { _ = P(%s) }", test.typ, test.val)
		types := make(map[ast.Expr]TypeAndValue)
		mustTypecheck(src, nil, &Info{Types: types})

		var n int
		for x, tv := range types {
			if x, _ := x.(*ast.CallExpr); x != nil {
				// there must be exactly one CallExpr which is the P(val) conversion
				n++
				tpar, _ := tv.Type.(*TypeParam)
				if tpar == nil {
					t.Fatalf("%s: got type %s, want type parameter", ExprString(x), tv.Type)
				}
				if name := tpar.Obj().Name(); name != "P" {
					t.Fatalf("%s: got type parameter name %s, want P", ExprString(x), name)
				}
				// P(val) must not be constant
				if tv.Value != nil {
					t.Errorf("%s: got constant value %s (%s), want no constant", ExprString(x), tv.Value, tv.Value.String())
				}
			}
		}

		if n != 1 {
			t.Fatalf("%s: got %d CallExpr nodes; want 1", src, 1)
		}
	}
}

func TestIssue54258(t *testing.T) {

	tests := []struct{ main, b, want string }{
		{ //---------------------------------------------------------------
			`package main
import "b"
type I0 interface {
	M0(w struct{ f string })
}
var _ I0 = b.S{}
`,
			`package b
type S struct{}
func (S) M0(struct{ f string }) {}
`,
			`6:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I0 value in variable declaration: b[.]S does not implement I0 [(]wrong type for method M0[)]
.*have M0[(]struct{f string /[*] package b [*]/ }[)]
.*want M0[(]struct{f string /[*] package main [*]/ }[)]`},

		{ //---------------------------------------------------------------
			`package main
import "b"
type I1 interface {
	M1(struct{ string })
}
var _ I1 = b.S{}
`,
			`package b
type S struct{}
func (S) M1(struct{ string }) {}
`,
			`6:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I1 value in variable declaration: b[.]S does not implement I1 [(]wrong type for method M1[)]
.*have M1[(]struct{string /[*] package b [*]/ }[)]
.*want M1[(]struct{string /[*] package main [*]/ }[)]`},

		{ //---------------------------------------------------------------
			`package main
import "b"
type I2 interface {
	M2(y struct{ f struct{ f string } })
}
var _ I2 = b.S{}
`,
			`package b
type S struct{}
func (S) M2(struct{ f struct{ f string } }) {}
`,
			`6:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I2 value in variable declaration: b[.]S does not implement I2 [(]wrong type for method M2[)]
.*have M2[(]struct{f struct{f string} /[*] package b [*]/ }[)]
.*want M2[(]struct{f struct{f string} /[*] package main [*]/ }[)]`},

		{ //---------------------------------------------------------------
			`package main
import "b"
type I3 interface {
	M3(z struct{ F struct{ f string } })
}
var _ I3 = b.S{}
`,
			`package b
type S struct{}
func (S) M3(struct{ F struct{ f string } }) {}
`,
			`6:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I3 value in variable declaration: b[.]S does not implement I3 [(]wrong type for method M3[)]
.*have M3[(]struct{F struct{f string /[*] package b [*]/ }}[)]
.*want M3[(]struct{F struct{f string /[*] package main [*]/ }}[)]`},

		{ //---------------------------------------------------------------
			`package main
import "b"
type I4 interface {
	M4(_ struct { *string })
}
var _ I4 = b.S{}
`,
			`package b
type S struct{}
func (S) M4(struct { *string }) {}
`,
			`6:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I4 value in variable declaration: b[.]S does not implement I4 [(]wrong type for method M4[)]
.*have M4[(]struct{[*]string /[*] package b [*]/ }[)]
.*want M4[(]struct{[*]string /[*] package main [*]/ }[)]`},

		{ //---------------------------------------------------------------
			`package main
import "b"
type t struct{ A int }
type I5 interface {
	M5(_ struct {b.S;t})
}
var _ I5 = b.S{}
`,
			`package b
type S struct{}
type t struct{ A int }
func (S) M5(struct {S;t}) {}
`,
			`7:12: cannot use b[.]S{} [(]value of type b[.]S[)] as I5 value in variable declaration: b[.]S does not implement I5 [(]wrong type for method M5[)]
.*have M5[(]struct{b[.]S; b[.]t}[)]
.*want M5[(]struct{b[.]S; t}[)]`},
	}

	fset := token.NewFileSet()
	test := func(main, b, want string) {
		re := regexp.MustCompile(want)
		bpkg := mustTypecheck(b, nil, nil)
		mast := mustParse(fset, main)
		conf := Config{Importer: importHelper{pkg: bpkg}}
		_, err := conf.Check(mast.Name.Name, fset, []*ast.File{mast}, nil)
		if err == nil {
			t.Error("Expected failure, but it did not")
		} else if got := err.Error(); !re.MatchString(got) {
			t.Errorf("Wanted match for\n\t%s\n but got\n\t%s", want, got)
		} else if testing.Verbose() {
			t.Logf("Saw expected\n\t%s", err.Error())
		}
	}
	for _, t := range tests {
		test(t.main, t.b, t.want)
	}
}

func TestIssue59944(t *testing.T) {
	testenv.MustHaveCGO(t)

	// The typechecker should resolve methods declared on aliases of cgo types.
	const src = `
package p

/*
struct layout {
	int field;
};
*/
import "C"

type Layout = C.struct_layout

func (l *Layout) Binding() {}

func _() {
	_ = (*Layout).Binding
}
`

	// code generated by cmd/cgo for the above source.
	const cgoTypes = `
// Code generated by cmd/cgo; DO NOT EDIT.

package p

import "unsafe"

import "syscall"

import _cgopackage "runtime/cgo"

type _ _cgopackage.Incomplete
var _ syscall.Errno
func _Cgo_ptr(ptr unsafe.Pointer) unsafe.Pointer { return ptr }

//go:linkname _Cgo_always_false runtime.cgoAlwaysFalse
var _Cgo_always_false bool
//go:linkname _Cgo_use runtime.cgoUse
func _Cgo_use(interface{})
type _Ctype_int int32

type _Ctype_struct_layout struct {
	field _Ctype_int
}

type _Ctype_void [0]byte

//go:linkname _cgo_runtime_cgocall runtime.cgocall
func _cgo_runtime_cgocall(unsafe.Pointer, uintptr) int32

//go:linkname _cgoCheckPointer runtime.cgoCheckPointer
func _cgoCheckPointer(interface{}, interface{})

//go:linkname _cgoCheckResult runtime.cgoCheckResult
func _cgoCheckResult(interface{})
`
	testFiles(t, []string{"p.go", "_cgo_gotypes.go"}, [][]byte{[]byte(src), []byte(cgoTypes)}, false, func(cfg *Config) {
		*boolFieldAddr(cfg, "go115UsesCgo") = true
	})
}

func TestIssue61931(t *testing.T) {
	const src = `
package p

func A(func(any), ...any) {}
func B[T any](T)          {}

func _() {
	A(B, nil // syntax error: missing ',' before newline in argument list
}
`
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, pkgName(src), src, 0)
	if err == nil {
		t.Fatal("expected syntax error")
	}

	var conf Config
	conf.Check(f.Name.Name, fset, []*ast.File{f}, nil) // must not panic
}

func TestIssue61938(t *testing.T) {
	const src = `
package p

func f[T any]() {}
func _()        { f() }
`
	// no error handler provided (this issue)
	var conf Config
	typecheck(src, &conf, nil) // must not panic

	// with error handler (sanity check)
	conf.Error = func(error) {}
	typecheck(src, &conf, nil) // must not panic
}

func TestIssue63260(t *testing.T) {
	const src = `
package p

func _() {
        use(f[*string])
}

func use(func()) {}

func f[I *T, T any]() {
        var v T
        _ = v
}`

	info := Info{
		Defs: make(map[*ast.Ident]Object),
	}
	pkg := mustTypecheck(src, nil, &info)

	// get type parameter T in signature of f
	T := pkg.Scope().Lookup("f").Type().(*Signature).TypeParams().At(1)
	if T.Obj().Name() != "T" {
		t.Fatalf("got type parameter %s, want T", T)
	}

	// get type of variable v in body of f
	var v Object
	for name, obj := range info.Defs {
		if name.Name == "v" {
			v = obj
			break
		}
	}
	if v == nil {
		t.Fatal("variable v not found")
	}

	// type of v and T must be pointer-identical
	if v.Type() != T {
		t.Fatalf("types of v and T are not pointer-identical: %p != %p", v.Type().(*TypeParam), T)
	}
}

func TestIssue44410(t *testing.T) {
	const src = `
package p

type A = []int
type S struct{ A }
`

	t.Setenv("GODEBUG", "gotypesalias=1")
	pkg := mustTypecheck(src, nil, nil)

	S := pkg.Scope().Lookup("S")
	if S == nil {
		t.Fatal("object S not found")
	}

	got := S.String()
	const want = "type p.S struct{p.A}"
	if got != want {
		t.Fatalf("got %q; want %q", got, want)
	}
}

func TestIssue59831(t *testing.T) {
	// Package a exports a type S with an unexported method m;
	// the tests check the error messages when m is not found.
	const asrc = `package a; type S struct{}; func (S) m() {}`
	apkg := mustTypecheck(asrc, nil, nil)

	// Package b exports a type S with an exported method m;
	// the tests check the error messages when M is not found.
	const bsrc = `package b; type S struct{}; func (S) M() {}`
	bpkg := mustTypecheck(bsrc, nil, nil)

	tests := []struct {
		imported *Package
		src, err string
	}{
		// tests importing a (or nothing)
		{apkg, `package a1; import "a"; var _ interface { M() } = a.S{}`,
			"a.S does not implement interface{M()} (missing method M) have m() want M()"},

		{apkg, `package a2; import "a"; var _ interface { m() } = a.S{}`,
			"a.S does not implement interface{m()} (unexported method m)"}, // test for issue

		{nil, `package a3; type S struct{}; func (S) m(); var _ interface { M() } = S{}`,
			"S does not implement interface{M()} (missing method M) have m() want M()"},

		{nil, `package a4; type S struct{}; func (S) m(); var _ interface { m() } = S{}`,
			""}, // no error expected

		{nil, `package a5; type S struct{}; func (S) m(); var _ interface { n() } = S{}`,
			"S does not implement interface{n()} (missing method n)"},

		// tests importing b (or nothing)
		{bpkg, `package b1; import "b"; var _ interface { m() } = b.S{}`,
			"b.S does not implement interface{m()} (missing method m) have M() want m()"},

		{bpkg, `package b2; import "b"; var _ interface { M() } = b.S{}`,
			""}, // no error expected

		{nil, `package b3; type S struct{}; func (S) M(); var _ interface { M() } = S{}`,
			""}, // no error expected

		{nil, `package b4; type S struct{}; func (S) M(); var _ interface { m() } = S{}`,
			"S does not implement interface{m()} (missing method m) have M() want m()"},

		{nil, `package b5; type S struct{}; func (S) M(); var _ interface { n() } = S{}`,
			"S does not implement interface{n()} (missing method n)"},
	}

	for _, test := range tests {
		// typecheck test source
		conf := Config{Importer: importHelper{pkg: test.imported}}
		pkg, err := typecheck(test.src, &conf, nil)
		if err == nil {
			if test.err != "" {
				t.Errorf("package %s: got no error, want %q", pkg.Name(), test.err)
			}
			continue
		}
		if test.err == "" {
			t.Errorf("package %s: got %q, want not error", pkg.Name(), err.Error())
		}

		// flatten reported error message
		errmsg := strings.ReplaceAll(err.Error(), "\n", " ")
		errmsg = strings.ReplaceAll(errmsg, "\t", "")

		// verify error message
		if !strings.Contains(errmsg, test.err) {
			t.Errorf("package %s: got %q, want %q", pkg.Name(), errmsg, test.err)
		}
	}
}

func TestIssue64759(t *testing.T) {
	const src = `
//go:build go1.18
package p

func f[S ~[]E, E any](S) {}

func _() {
	f([]string{})
}
`
	// Per the go:build directive, the source must typecheck
	// even though the (module) Go version is set to go1.17.
	conf := Config{GoVersion: "go1.17"}
	mustTypecheck(src, &conf, nil)
}
