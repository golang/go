// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectlite_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
)

var typeNames = []string{
	"uncommonType",
	"arrayType",
	"chanType",
	"funcType",
	"interfaceType",
	"mapType",
	"ptrType",
	"sliceType",
	"structType",
}

type visitor struct {
	m map[string]map[string]bool
}

func newVisitor() visitor {
	v := visitor{}
	v.m = make(map[string]map[string]bool)

	return v
}
func (v visitor) filter(name string) bool {
	for _, typeName := range typeNames {
		if typeName == name {
			return true
		}
	}
	return false
}

func (v visitor) Visit(n ast.Node) ast.Visitor {
	switch x := n.(type) {
	case *ast.TypeSpec:
		if v.filter(x.Name.String()) {
			if st, ok := x.Type.(*ast.StructType); ok {
				v.m[x.Name.String()] = make(map[string]bool)
				for _, field := range st.Fields.List {
					k := fmt.Sprintf("%s", field.Type)
					if len(field.Names) > 0 {
						k = field.Names[0].Name
					}
					v.m[x.Name.String()][k] = true
				}
			}
		}
	}
	return v
}

func loadTypes(path, pkgName string, v visitor) {
	fset := token.NewFileSet()

	filter := func(fi fs.FileInfo) bool {
		return strings.HasSuffix(fi.Name(), ".go")
	}
	pkgs, err := parser.ParseDir(fset, path, filter, 0)
	if err != nil {
		panic(err)
	}

	pkg := pkgs[pkgName]

	for _, f := range pkg.Files {
		ast.Walk(v, f)
	}
}

func TestMirrorWithReflect(t *testing.T) {
	// TODO when the dust clears, figure out what this should actually test.
	t.Skipf("reflect and reflectlite are out of sync for now")
	reflectDir := filepath.Join(runtime.GOROOT(), "src", "reflect")
	if _, err := os.Stat(reflectDir); os.IsNotExist(err) {
		// On some mobile builders, the test binary executes on a machine without a
		// complete GOROOT source tree.
		t.Skipf("GOROOT source not present")
	}

	var wg sync.WaitGroup
	rl, r := newVisitor(), newVisitor()

	for _, tc := range []struct {
		path, pkg string
		v         visitor
	}{
		{".", "reflectlite", rl},
		{reflectDir, "reflect", r},
	} {
		tc := tc
		wg.Add(1)
		go func() {
			defer wg.Done()
			loadTypes(tc.path, tc.pkg, tc.v)
		}()
	}
	wg.Wait()

	if len(rl.m) != len(r.m) {
		t.Fatalf("number of types mismatch, reflect: %d, reflectlite: %d (%+v, %+v)", len(r.m), len(rl.m), r.m, rl.m)
	}

	for typName := range r.m {
		if len(r.m[typName]) != len(rl.m[typName]) {
			t.Errorf("type %s number of fields mismatch, reflect: %d, reflectlite: %d", typName, len(r.m[typName]), len(rl.m[typName]))
			continue
		}
		for field := range r.m[typName] {
			if _, ok := rl.m[typName][field]; !ok {
				t.Errorf(`Field mismatch, reflect have "%s", relectlite does not.`, field)
			}
		}
	}
}
