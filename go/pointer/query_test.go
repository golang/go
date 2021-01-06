// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pointer

import (
	"reflect"
	"testing"

	"golang.org/x/tools/go/loader"
)

func TestParseExtendedQuery(t *testing.T) {
	const myprog = `
package pkg

import "reflect"

type T []*int

var V1 *int
var V2 **int
var V3 []*int
var V4 chan []*int
var V5 struct {F1, F2 chan *int}
var V6 [1]chan *int
var V7 int
var V8 T
var V9 reflect.Value
`
	tests := []struct {
		in    string
		out   []interface{}
		v     string
		valid bool
	}{
		{`x`, []interface{}{"x"}, "V1", true},
		{`x`, []interface{}{"x"}, "V9", true},
		{`*x`, []interface{}{"x", "load"}, "V2", true},
		{`x[0]`, []interface{}{"x", "sliceelem"}, "V3", true},
		{`x[0]`, []interface{}{"x", "sliceelem"}, "V8", true},
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

	var conf loader.Config
	f, err := conf.ParseFile("file.go", myprog)
	if err != nil {
		t.Fatal(err)
	}
	conf.CreateFromFiles("main", f)
	lprog, err := conf.Load()
	if err != nil {
		t.Fatal(err)
	}
	pkg := lprog.Created[0].Pkg

	for _, test := range tests {
		typ := pkg.Scope().Lookup(test.v).Type()
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
