package typeutil_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	"golang.org/x/tools/go/types"
	"golang.org/x/tools/go/types/typeutil"
)

func TestDependencies(t *testing.T) {
	packages := make(map[string]*types.Package)
	conf := types.Config{
		Packages: packages,
		Import: func(_ map[string]*types.Package, path string) (*types.Package, error) {
			return packages[path], nil
		},
	}
	fset := token.NewFileSet()

	// All edges go to the right.
	//  /--D--B--A
	// F    \_C_/
	//  \__E_/
	for i, content := range []string{
		`package A`,
		`package C; import (_ "A")`,
		`package B; import (_ "A")`,
		`package E; import (_ "C")`,
		`package D; import (_ "B"; _ "C")`,
		`package F; import (_ "D"; _ "E")`,
	} {
		f, err := parser.ParseFile(fset, fmt.Sprintf("%d.go", i), content, 0)
		if err != nil {
			t.Fatal(err)
		}
		pkg, err := conf.Check(f.Name.Name, fset, []*ast.File{f}, nil)
		if err != nil {
			t.Fatal(err)
		}
		packages[pkg.Path()] = pkg
	}

	for _, test := range []struct {
		roots, want string
	}{
		{"A", "A"},
		{"B", "AB"},
		{"C", "AC"},
		{"D", "ABCD"},
		{"E", "ACE"},
		{"F", "ABCDEF"},

		{"BE", "ABCE"},
		{"EB", "ACEB"},
		{"DE", "ABCDE"},
		{"ED", "ACEBD"},
		{"EF", "ACEBDF"},
	} {
		var pkgs []*types.Package
		for _, r := range test.roots {
			pkgs = append(pkgs, conf.Packages[string(r)])
		}
		var got string
		for _, p := range typeutil.Dependencies(pkgs...) {
			got += p.Path()
		}
		if got != test.want {
			t.Errorf("Dependencies(%q) = %q, want %q", test.roots, got, test.want)
		}
	}
}
