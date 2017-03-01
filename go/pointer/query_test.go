package pointer

import (
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"reflect"
	"testing"
)

func TestParseExtendedQuery(t *testing.T) {
	const myprog = `
package pkg
var V1 *int
var V2 **int
var V3 []*int
var V4 chan []*int
var V5 struct {F1, F2 chan *int}
var V6 [1]chan *int
var V7 int
`
	tests := []struct {
		in    string
		out   []interface{}
		v     string
		valid bool
	}{
		{`x`, []interface{}{"x"}, "V1", true},
		{`*x`, []interface{}{"x", "load"}, "V2", true},
		{`x[0]`, []interface{}{"x", "sliceelem"}, "V3", true},
		{`<-x`, []interface{}{"x", "recv"}, "V4", true},
		{`(<-x)[0]`, []interface{}{"x", "recv", "sliceelem"}, "V4", true},
		{`<-x.F2`, []interface{}{"x", "field", 1, "recv"}, "V5", true},
		{`<-x[0]`, []interface{}{"x", "arrayelem", "recv"}, "V6", true},
		{`x`, nil, "V7", false},
		{`y`, nil, "V1", false},
		{`x; x`, nil, "V1", false},
		{`x()`, nil, "V1", false},
		{`close(x)`, nil, "V1", false},
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "file.go", myprog, 0)
	if err != nil {
		t.Fatal(err)
	}
	cfg := &types.Config{}
	pkg, err := cfg.Check("main", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, test := range tests {
		typ := pkg.Scope().Lookup(test.v).Type().Underlying()
		ops, _, err := parseExtendedQuery(typ, test.in)
		if test.valid && err != nil {
			t.Errorf("parseExtendedQuery(%q) = %s, expected no error", test.in, err)
		}
		if !test.valid && err == nil {
			t.Errorf("parseExtendedQuery(%q) succeeded, expected error", test.in)
		}

		if !reflect.DeepEqual(ops, test.out) {
			t.Errorf("parseExtendedQuery(%q) = %#v, want %#v", test.in, ops, test.out)
		}
	}
}
