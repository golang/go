// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"go/ast"
	"go/build"
	"go/importer"
	"go/parser"
	"go/printer"
	"go/token"
	"go/types"
	"os"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

// Check that 64-bit fields on which we apply atomic operations
// are aligned to 8 bytes. This can be a problem on 32-bit systems.
func TestAtomicAlignment(t *testing.T) {
	// Read the code making the tables above, to see which fields and
	// variables we are currently checking.
	checked := map[string]bool{}
	x, err := os.ReadFile("./align_runtime_test.go")
	if err != nil {
		t.Fatalf("read failed: %v", err)
	}
	fieldDesc := map[int]string{}
	r := regexp.MustCompile(`unsafe[.]Offsetof[(](\w+){}[.](\w+)[)]`)
	matches := r.FindAllStringSubmatch(string(x), -1)
	for i, v := range matches {
		checked["field runtime."+v[1]+"."+v[2]] = true
		fieldDesc[i] = v[1] + "." + v[2]
	}
	varDesc := map[int]string{}
	r = regexp.MustCompile(`unsafe[.]Pointer[(]&(\w+)[)]`)
	matches = r.FindAllStringSubmatch(string(x), -1)
	for i, v := range matches {
		checked["var "+v[1]] = true
		varDesc[i] = v[1]
	}

	// Check all of our alignemnts. This is the actual core of the test.
	for i, d := range runtime.AtomicFields {
		if d%8 != 0 {
			t.Errorf("field alignment of %s failed: offset is %d", fieldDesc[i], d)
		}
	}
	for i, p := range runtime.AtomicVariables {
		if uintptr(p)%8 != 0 {
			t.Errorf("variable alignment of %s failed: address is %x", varDesc[i], p)
		}
	}

	// The code above is the actual test. The code below attempts to check
	// that the tables used by the code above are exhaustive.

	// Parse the whole runtime package, checking that arguments of
	// appropriate atomic operations are in the list above.
	fset := token.NewFileSet()
	m, err := parser.ParseDir(fset, ".", nil, 0)
	if err != nil {
		t.Fatalf("parsing runtime failed: %v", err)
	}
	pkg := m["runtime"] // Note: ignore runtime_test and main packages

	// Filter files by those for the current architecture/os being tested.
	fileMap := map[string]bool{}
	for _, f := range buildableFiles(t, ".") {
		fileMap[f] = true
	}
	var files []*ast.File
	for fname, f := range pkg.Files {
		if fileMap[fname] {
			files = append(files, f)
		}
	}

	// Call go/types to analyze the runtime package.
	var info types.Info
	info.Types = map[ast.Expr]types.TypeAndValue{}
	conf := types.Config{Importer: importer.Default()}
	_, err = conf.Check("runtime", fset, files, &info)
	if err != nil {
		t.Fatalf("typechecking runtime failed: %v", err)
	}

	// Analyze all atomic.*64 callsites.
	v := Visitor{t: t, fset: fset, types: info.Types, checked: checked}
	ast.Walk(&v, pkg)
}

type Visitor struct {
	fset    *token.FileSet
	types   map[ast.Expr]types.TypeAndValue
	checked map[string]bool
	t       *testing.T
}

func (v *Visitor) Visit(n ast.Node) ast.Visitor {
	c, ok := n.(*ast.CallExpr)
	if !ok {
		return v
	}
	f, ok := c.Fun.(*ast.SelectorExpr)
	if !ok {
		return v
	}
	p, ok := f.X.(*ast.Ident)
	if !ok {
		return v
	}
	if p.Name != "atomic" {
		return v
	}
	if !strings.HasSuffix(f.Sel.Name, "64") {
		return v
	}

	a := c.Args[0]

	// This is a call to atomic.XXX64(a, ...). Make sure a is aligned to 8 bytes.
	// XXX = one of Load, Store, Cas, etc.
	// The arg we care about the alignment of is always the first one.

	if u, ok := a.(*ast.UnaryExpr); ok && u.Op == token.AND {
		v.checkAddr(u.X)
		return v
	}

	// Other cases there's nothing we can check. Assume we're ok.
	v.t.Logf("unchecked atomic operation %s %v", v.fset.Position(n.Pos()), v.print(n))

	return v
}

// checkAddr checks to make sure n is a properly aligned address for a 64-bit atomic operation.
func (v *Visitor) checkAddr(n ast.Node) {
	switch n := n.(type) {
	case *ast.IndexExpr:
		// Alignment of an array element is the same as the whole array.
		v.checkAddr(n.X)
		return
	case *ast.Ident:
		key := "var " + v.print(n)
		if !v.checked[key] {
			v.t.Errorf("unchecked variable %s %s", v.fset.Position(n.Pos()), key)
		}
		return
	case *ast.SelectorExpr:
		t := v.types[n.X].Type
		if t == nil {
			// Not sure what is happening here, go/types fails to
			// type the selector arg on some platforms.
			return
		}
		if p, ok := t.(*types.Pointer); ok {
			// Note: we assume here that the pointer p in p.foo is properly
			// aligned. We just check that foo is at a properly aligned offset.
			t = p.Elem()
		} else {
			v.checkAddr(n.X)
		}
		if t.Underlying() == t {
			v.t.Errorf("analysis can't handle unnamed type %s %v", v.fset.Position(n.Pos()), t)
		}
		key := "field " + t.String() + "." + n.Sel.Name
		if !v.checked[key] {
			v.t.Errorf("unchecked field %s %s", v.fset.Position(n.Pos()), key)
		}
	default:
		v.t.Errorf("unchecked atomic address %s %v", v.fset.Position(n.Pos()), v.print(n))

	}
}

func (v *Visitor) print(n ast.Node) string {
	var b strings.Builder
	printer.Fprint(&b, v.fset, n)
	return b.String()
}

// buildableFiles returns the list of files in the given directory
// that are actually used for the build, given GOOS/GOARCH restrictions.
func buildableFiles(t *testing.T, dir string) []string {
	ctxt := build.Default
	ctxt.CgoEnabled = true
	pkg, err := ctxt.ImportDir(dir, 0)
	if err != nil {
		t.Fatalf("can't find buildable files: %v", err)
	}
	return pkg.GoFiles
}
