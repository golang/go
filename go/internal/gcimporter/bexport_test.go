// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter_test

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/constant"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/internal/gcimporter"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/internal/typeparams"
	"golang.org/x/tools/internal/typeparams/genericfeatures"
)

var isRace = false

func TestBExportData_stdlib(t *testing.T) {
	if runtime.Compiler == "gccgo" {
		t.Skip("gccgo standard library is inaccessible")
	}
	if runtime.GOOS == "android" {
		t.Skipf("incomplete std lib on %s", runtime.GOOS)
	}
	if isRace {
		t.Skipf("stdlib tests take too long in race mode and flake on builders")
	}
	if testing.Short() {
		t.Skip("skipping RAM hungry test in -short mode")
	}

	// Load, parse and type-check the program.
	ctxt := build.Default // copy
	ctxt.GOPATH = ""      // disable GOPATH
	conf := loader.Config{
		Build:       &ctxt,
		AllowErrors: true,
		TypeChecker: types.Config{
			Error: func(err error) { t.Log(err) },
		},
	}
	for _, path := range buildutil.AllPackages(conf.Build) {
		conf.Import(path)
	}

	// Create a package containing type and value errors to ensure
	// they are properly encoded/decoded.
	f, err := conf.ParseFile("haserrors/haserrors.go", `package haserrors
const UnknownValue = "" + 0
type UnknownType undefined
`)
	if err != nil {
		t.Fatal(err)
	}
	conf.CreateFromFiles("haserrors", f)

	prog, err := conf.Load()
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	numPkgs := len(prog.AllPackages)
	if want := minStdlibPackages; numPkgs < want {
		t.Errorf("Loaded only %d packages, want at least %d", numPkgs, want)
	}

	checked := 0
	for pkg, info := range prog.AllPackages {
		if info.Files == nil {
			continue // empty directory
		}
		// Binary export does not support generic code.
		inspect := inspector.New(info.Files)
		if genericfeatures.ForPackage(inspect, &info.Info) != 0 {
			t.Logf("skipping package %q which uses generics", pkg.Path())
			continue
		}
		checked++
		exportdata, err := gcimporter.BExportData(conf.Fset, pkg)
		if err != nil {
			t.Fatal(err)
		}

		imports := make(map[string]*types.Package)
		fset2 := token.NewFileSet()
		n, pkg2, err := gcimporter.BImportData(fset2, imports, exportdata, pkg.Path())
		if err != nil {
			t.Errorf("BImportData(%s): %v", pkg.Path(), err)
			continue
		}
		if n != len(exportdata) {
			t.Errorf("BImportData(%s) decoded %d bytes, want %d",
				pkg.Path(), n, len(exportdata))
		}

		// Compare the packages' corresponding members.
		for _, name := range pkg.Scope().Names() {
			if !ast.IsExported(name) {
				continue
			}
			obj1 := pkg.Scope().Lookup(name)
			obj2 := pkg2.Scope().Lookup(name)
			if obj2 == nil {
				t.Errorf("%s.%s not found, want %s", pkg.Path(), name, obj1)
				continue
			}

			fl1 := fileLine(conf.Fset, obj1)
			fl2 := fileLine(fset2, obj2)
			if fl1 != fl2 {
				t.Errorf("%s.%s: got posn %s, want %s",
					pkg.Path(), name, fl2, fl1)
			}

			if err := equalObj(obj1, obj2); err != nil {
				t.Errorf("%s.%s: %s\ngot:  %s\nwant: %s",
					pkg.Path(), name, err, obj2, obj1)
			}
		}
	}
	if want := minStdlibPackages; checked < want {
		t.Errorf("Checked only %d packages, want at least %d", checked, want)
	}
}

func fileLine(fset *token.FileSet, obj types.Object) string {
	posn := fset.Position(obj.Pos())
	filename := filepath.Clean(strings.ReplaceAll(posn.Filename, "$GOROOT", runtime.GOROOT()))
	return fmt.Sprintf("%s:%d", filename, posn.Line)
}

// equalObj reports how x and y differ.  They are assumed to belong to
// different universes so cannot be compared directly.
func equalObj(x, y types.Object) error {
	if reflect.TypeOf(x) != reflect.TypeOf(y) {
		return fmt.Errorf("%T vs %T", x, y)
	}
	xt := x.Type()
	yt := y.Type()
	switch x.(type) {
	case *types.Var, *types.Func:
		// ok
	case *types.Const:
		xval := x.(*types.Const).Val()
		yval := y.(*types.Const).Val()
		// Use string comparison for floating-point values since rounding is permitted.
		if constant.Compare(xval, token.NEQ, yval) &&
			!(xval.Kind() == constant.Float && xval.String() == yval.String()) {
			return fmt.Errorf("unequal constants %s vs %s", xval, yval)
		}
	case *types.TypeName:
		xt = xt.Underlying()
		yt = yt.Underlying()
	default:
		return fmt.Errorf("unexpected %T", x)
	}
	return equalType(xt, yt)
}

func equalType(x, y types.Type) error {
	if reflect.TypeOf(x) != reflect.TypeOf(y) {
		return fmt.Errorf("unequal kinds: %T vs %T", x, y)
	}
	switch x := x.(type) {
	case *types.Interface:
		y := y.(*types.Interface)
		// TODO(gri): enable separate emission of Embedded interfaces
		// and ExplicitMethods then use this logic.
		// if x.NumEmbeddeds() != y.NumEmbeddeds() {
		// 	return fmt.Errorf("unequal number of embedded interfaces: %d vs %d",
		// 		x.NumEmbeddeds(), y.NumEmbeddeds())
		// }
		// for i := 0; i < x.NumEmbeddeds(); i++ {
		// 	xi := x.Embedded(i)
		// 	yi := y.Embedded(i)
		// 	if xi.String() != yi.String() {
		// 		return fmt.Errorf("mismatched %th embedded interface: %s vs %s",
		// 			i, xi, yi)
		// 	}
		// }
		// if x.NumExplicitMethods() != y.NumExplicitMethods() {
		// 	return fmt.Errorf("unequal methods: %d vs %d",
		// 		x.NumExplicitMethods(), y.NumExplicitMethods())
		// }
		// for i := 0; i < x.NumExplicitMethods(); i++ {
		// 	xm := x.ExplicitMethod(i)
		// 	ym := y.ExplicitMethod(i)
		// 	if xm.Name() != ym.Name() {
		// 		return fmt.Errorf("mismatched %th method: %s vs %s", i, xm, ym)
		// 	}
		// 	if err := equalType(xm.Type(), ym.Type()); err != nil {
		// 		return fmt.Errorf("mismatched %s method: %s", xm.Name(), err)
		// 	}
		// }
		if x.NumMethods() != y.NumMethods() {
			return fmt.Errorf("unequal methods: %d vs %d",
				x.NumMethods(), y.NumMethods())
		}
		for i := 0; i < x.NumMethods(); i++ {
			xm := x.Method(i)
			ym := y.Method(i)
			if xm.Name() != ym.Name() {
				return fmt.Errorf("mismatched %dth method: %s vs %s", i, xm, ym)
			}
			if err := equalType(xm.Type(), ym.Type()); err != nil {
				return fmt.Errorf("mismatched %s method: %s", xm.Name(), err)
			}
		}
		// Constraints are handled explicitly in the *TypeParam case below, so we
		// don't yet need to consider embeddeds here.
		// TODO(rfindley): consider the type set here.
	case *types.Array:
		y := y.(*types.Array)
		if x.Len() != y.Len() {
			return fmt.Errorf("unequal array lengths: %d vs %d", x.Len(), y.Len())
		}
		if err := equalType(x.Elem(), y.Elem()); err != nil {
			return fmt.Errorf("array elements: %s", err)
		}
	case *types.Basic:
		y := y.(*types.Basic)
		if x.Kind() != y.Kind() {
			return fmt.Errorf("unequal basic types: %s vs %s", x, y)
		}
	case *types.Chan:
		y := y.(*types.Chan)
		if x.Dir() != y.Dir() {
			return fmt.Errorf("unequal channel directions: %d vs %d", x.Dir(), y.Dir())
		}
		if err := equalType(x.Elem(), y.Elem()); err != nil {
			return fmt.Errorf("channel elements: %s", err)
		}
	case *types.Map:
		y := y.(*types.Map)
		if err := equalType(x.Key(), y.Key()); err != nil {
			return fmt.Errorf("map keys: %s", err)
		}
		if err := equalType(x.Elem(), y.Elem()); err != nil {
			return fmt.Errorf("map values: %s", err)
		}
	case *types.Named:
		y := y.(*types.Named)
		xOrig := typeparams.NamedTypeOrigin(x)
		yOrig := typeparams.NamedTypeOrigin(y)
		if sanitizeName(xOrig.String()) != sanitizeName(yOrig.String()) {
			return fmt.Errorf("unequal named types: %s vs %s", x, y)
		}
		if err := equalTypeArgs(typeparams.NamedTypeArgs(x), typeparams.NamedTypeArgs(y)); err != nil {
			return fmt.Errorf("type arguments: %s", err)
		}
		if x.NumMethods() != y.NumMethods() {
			return fmt.Errorf("unequal methods: %d vs %d",
				x.NumMethods(), y.NumMethods())
		}
		// Unfortunately method sorting is not canonical, so sort before comparing.
		var xms, yms []*types.Func
		for i := 0; i < x.NumMethods(); i++ {
			xms = append(xms, x.Method(i))
			yms = append(yms, y.Method(i))
		}
		for _, ms := range [][]*types.Func{xms, yms} {
			sort.Slice(ms, func(i, j int) bool {
				return ms[i].Name() < ms[j].Name()
			})
		}
		for i, xm := range xms {
			ym := yms[i]
			if xm.Name() != ym.Name() {
				return fmt.Errorf("mismatched %dth method: %s vs %s", i, xm, ym)
			}
			// Calling equalType here leads to infinite recursion, so just compare
			// strings.
			if sanitizeName(xm.String()) != sanitizeName(ym.String()) {
				return fmt.Errorf("unequal methods: %s vs %s", x, y)
			}
		}
	case *types.Pointer:
		y := y.(*types.Pointer)
		if err := equalType(x.Elem(), y.Elem()); err != nil {
			return fmt.Errorf("pointer elements: %s", err)
		}
	case *types.Signature:
		y := y.(*types.Signature)
		if err := equalType(x.Params(), y.Params()); err != nil {
			return fmt.Errorf("parameters: %s", err)
		}
		if err := equalType(x.Results(), y.Results()); err != nil {
			return fmt.Errorf("results: %s", err)
		}
		if x.Variadic() != y.Variadic() {
			return fmt.Errorf("unequal variadicity: %t vs %t",
				x.Variadic(), y.Variadic())
		}
		if (x.Recv() != nil) != (y.Recv() != nil) {
			return fmt.Errorf("unequal receivers: %s vs %s", x.Recv(), y.Recv())
		}
		if x.Recv() != nil {
			// TODO(adonovan): fix: this assertion fires for interface methods.
			// The type of the receiver of an interface method is a named type
			// if the Package was loaded from export data, or an unnamed (interface)
			// type if the Package was produced by type-checking ASTs.
			// if err := equalType(x.Recv().Type(), y.Recv().Type()); err != nil {
			// 	return fmt.Errorf("receiver: %s", err)
			// }
		}
		if err := equalTypeParams(typeparams.ForSignature(x), typeparams.ForSignature(y)); err != nil {
			return fmt.Errorf("type params: %s", err)
		}
		if err := equalTypeParams(typeparams.RecvTypeParams(x), typeparams.RecvTypeParams(y)); err != nil {
			return fmt.Errorf("recv type params: %s", err)
		}
	case *types.Slice:
		y := y.(*types.Slice)
		if err := equalType(x.Elem(), y.Elem()); err != nil {
			return fmt.Errorf("slice elements: %s", err)
		}
	case *types.Struct:
		y := y.(*types.Struct)
		if x.NumFields() != y.NumFields() {
			return fmt.Errorf("unequal struct fields: %d vs %d",
				x.NumFields(), y.NumFields())
		}
		for i := 0; i < x.NumFields(); i++ {
			xf := x.Field(i)
			yf := y.Field(i)
			if xf.Name() != yf.Name() {
				return fmt.Errorf("mismatched fields: %s vs %s", xf, yf)
			}
			if err := equalType(xf.Type(), yf.Type()); err != nil {
				return fmt.Errorf("struct field %s: %s", xf.Name(), err)
			}
			if x.Tag(i) != y.Tag(i) {
				return fmt.Errorf("struct field %s has unequal tags: %q vs %q",
					xf.Name(), x.Tag(i), y.Tag(i))
			}
		}
	case *types.Tuple:
		y := y.(*types.Tuple)
		if x.Len() != y.Len() {
			return fmt.Errorf("unequal tuple lengths: %d vs %d", x.Len(), y.Len())
		}
		for i := 0; i < x.Len(); i++ {
			if err := equalType(x.At(i).Type(), y.At(i).Type()); err != nil {
				return fmt.Errorf("tuple element %d: %s", i, err)
			}
		}
	case *typeparams.TypeParam:
		y := y.(*typeparams.TypeParam)
		if sanitizeName(x.String()) != sanitizeName(y.String()) {
			return fmt.Errorf("unequal named types: %s vs %s", x, y)
		}
		// For now, just compare constraints by type string to short-circuit
		// cycles. We have to make interfaces explicit as export data currently
		// doesn't support marking interfaces as implicit.
		// TODO(rfindley): remove makeExplicit once export data contains an
		// implicit bit.
		xc := sanitizeName(makeExplicit(x.Constraint()).String())
		yc := sanitizeName(makeExplicit(y.Constraint()).String())
		if xc != yc {
			return fmt.Errorf("unequal constraints: %s vs %s", xc, yc)
		}

	default:
		panic(fmt.Sprintf("unexpected %T type", x))
	}
	return nil
}

// makeExplicit returns an explicit version of typ, if typ is an implicit
// interface. Otherwise it returns typ unmodified.
func makeExplicit(typ types.Type) types.Type {
	if iface, _ := typ.(*types.Interface); iface != nil && typeparams.IsImplicit(iface) {
		var methods []*types.Func
		for i := 0; i < iface.NumExplicitMethods(); i++ {
			methods = append(methods, iface.Method(i))
		}
		var embeddeds []types.Type
		for i := 0; i < iface.NumEmbeddeds(); i++ {
			embeddeds = append(embeddeds, iface.EmbeddedType(i))
		}
		return types.NewInterfaceType(methods, embeddeds)
	}
	return typ
}

func equalTypeArgs(x, y *typeparams.TypeList) error {
	if x.Len() != y.Len() {
		return fmt.Errorf("unequal lengths: %d vs %d", x.Len(), y.Len())
	}
	for i := 0; i < x.Len(); i++ {
		if err := equalType(x.At(i), y.At(i)); err != nil {
			return fmt.Errorf("type %d: %s", i, err)
		}
	}
	return nil
}

func equalTypeParams(x, y *typeparams.TypeParamList) error {
	if x.Len() != y.Len() {
		return fmt.Errorf("unequal lengths: %d vs %d", x.Len(), y.Len())
	}
	for i := 0; i < x.Len(); i++ {
		if err := equalType(x.At(i), y.At(i)); err != nil {
			return fmt.Errorf("type parameter %d: %s", i, err)
		}
	}
	return nil
}

// sanitizeName removes type parameter debugging markers from an object
// or type string, to normalize it for comparison.
// TODO(rfindley): remove once this is no longer necessary.
func sanitizeName(name string) string {
	var runes []rune
	for _, r := range name {
		if '₀' <= r && r < '₀'+10 {
			continue // trim type parameter subscripts
		}
		runes = append(runes, r)
	}
	return string(runes)
}

// TestVeryLongFile tests the position of an import object declared in
// a very long input file.  Line numbers greater than maxlines are
// reported as line 1, not garbage or token.NoPos.
func TestVeryLongFile(t *testing.T) {
	// parse and typecheck
	longFile := "package foo" + strings.Repeat("\n", 123456) + "var X int"
	fset1 := token.NewFileSet()
	f, err := parser.ParseFile(fset1, "foo.go", longFile, 0)
	if err != nil {
		t.Fatal(err)
	}
	var conf types.Config
	pkg, err := conf.Check("foo", fset1, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// export
	exportdata, err := gcimporter.BExportData(fset1, pkg)
	if err != nil {
		t.Fatal(err)
	}

	// import
	imports := make(map[string]*types.Package)
	fset2 := token.NewFileSet()
	_, pkg2, err := gcimporter.BImportData(fset2, imports, exportdata, pkg.Path())
	if err != nil {
		t.Fatalf("BImportData(%s): %v", pkg.Path(), err)
	}

	// compare
	posn1 := fset1.Position(pkg.Scope().Lookup("X").Pos())
	posn2 := fset2.Position(pkg2.Scope().Lookup("X").Pos())
	if want := "foo.go:1:1"; posn2.String() != want {
		t.Errorf("X position = %s, want %s (orig was %s)",
			posn2, want, posn1)
	}
}

const src = `
package p

type (
	T0 = int32
	T1 = struct{}
	T2 = struct{ T1 }
	Invalid = foo // foo is undeclared
)
`

func checkPkg(t *testing.T, pkg *types.Package, label string) {
	T1 := types.NewStruct(nil, nil)
	T2 := types.NewStruct([]*types.Var{types.NewField(0, pkg, "T1", T1, true)}, nil)

	for _, test := range []struct {
		name string
		typ  types.Type
	}{
		{"T0", types.Typ[types.Int32]},
		{"T1", T1},
		{"T2", T2},
		{"Invalid", types.Typ[types.Invalid]},
	} {
		obj := pkg.Scope().Lookup(test.name)
		if obj == nil {
			t.Errorf("%s: %s not found", label, test.name)
			continue
		}
		tname, _ := obj.(*types.TypeName)
		if tname == nil {
			t.Errorf("%s: %v not a type name", label, obj)
			continue
		}
		if !tname.IsAlias() {
			t.Errorf("%s: %v: not marked as alias", label, tname)
			continue
		}
		if got := tname.Type(); !types.Identical(got, test.typ) {
			t.Errorf("%s: %v: got %v; want %v", label, tname, got, test.typ)
		}
	}
}

func TestTypeAliases(t *testing.T) {
	// parse and typecheck
	fset1 := token.NewFileSet()
	f, err := parser.ParseFile(fset1, "p.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	var conf types.Config
	pkg1, err := conf.Check("p", fset1, []*ast.File{f}, nil)
	if err == nil {
		// foo in undeclared in src; we should see an error
		t.Fatal("invalid source type-checked without error")
	}
	if pkg1 == nil {
		// despite incorrect src we should see a (partially) type-checked package
		t.Fatal("nil package returned")
	}
	checkPkg(t, pkg1, "export")

	// export
	exportdata, err := gcimporter.BExportData(fset1, pkg1)
	if err != nil {
		t.Fatal(err)
	}

	// import
	imports := make(map[string]*types.Package)
	fset2 := token.NewFileSet()
	_, pkg2, err := gcimporter.BImportData(fset2, imports, exportdata, pkg1.Path())
	if err != nil {
		t.Fatalf("BImportData(%s): %v", pkg1.Path(), err)
	}
	checkPkg(t, pkg2, "import")
}
