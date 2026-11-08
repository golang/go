// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

const filterTestSrc = `// Package p exercises Package.Filter.
package p

const KeepConst = 1
const DropConst = 2

var KeepVar = 1
var DropVar = 2

func KeepFunc() {}
func DropFunc() {}

// KeepType matches the filter by name, so it and all of its
// members are kept.
type KeepType struct{ Field int }

func (KeepType) KeepMethod() {}
func (KeepType) DropMethod() {}

// DropType matches nothing and is dropped entirely.
type DropType struct{ Other int }

func (DropType) DropMethod() {}

// PartialType matches only through one of its methods, so it is
// kept with just the matching members.
type PartialType struct{ Other int }

func (PartialType) KeepMethod() {}
func (PartialType) DropMethod() {}
`

func filterTestPackage(t *testing.T) *Package {
	t.Helper()
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "p.go", filterTestSrc, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}
	p, err := NewFromFiles(fset, []*ast.File{f}, "p", AllDecls)
	if err != nil {
		t.Fatal(err)
	}
	return p
}

func valueNames(values []*Value) []string {
	var names []string
	for _, v := range values {
		names = append(names, v.Names...)
	}
	return names
}

func funcNames(funcs []*Func) []string {
	var names []string
	for _, f := range funcs {
		names = append(names, f.Name)
	}
	return names
}

func typeNames(types []*Type) []string {
	var names []string
	for _, t := range types {
		names = append(names, t.Name)
	}
	return names
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFilter(t *testing.T) {
	p := filterTestPackage(t)
	if p.Doc == "" {
		t.Fatal("test package has no doc comment")
	}
	keep := map[string]bool{
		"KeepConst": true,
		"KeepVar":   true,
		"KeepFunc":  true,
		"KeepType":  true,
	}
	p.Filter(func(name string) bool {
		if name == "KeepMethod" {
			return true
		}
		return keep[name]
	})

	if p.Doc != "" {
		t.Errorf("after Filter, Doc = %q, want empty", p.Doc)
	}
	if got, want := valueNames(p.Consts), []string{"KeepConst"}; !equalStrings(got, want) {
		t.Errorf("after Filter, Consts = %v, want %v", got, want)
	}
	if got, want := valueNames(p.Vars), []string{"KeepVar"}; !equalStrings(got, want) {
		t.Errorf("after Filter, Vars = %v, want %v", got, want)
	}
	if got, want := funcNames(p.Funcs), []string{"KeepFunc"}; !equalStrings(got, want) {
		t.Errorf("after Filter, Funcs = %v, want %v", got, want)
	}
	if got, want := typeNames(p.Types), []string{"KeepType", "PartialType"}; !equalStrings(got, want) {
		t.Fatalf("after Filter, Types = %v, want %v", got, want)
	}
	// A type matching by name keeps all of its members.
	if got, want := funcNames(p.Types[0].Methods), []string{"DropMethod", "KeepMethod"}; !equalStrings(got, want) {
		t.Errorf("after Filter, KeepType.Methods = %v, want %v", got, want)
	}
	// A type matching only by member keeps just the matching members.
	if got, want := funcNames(p.Types[1].Methods), []string{"KeepMethod"}; !equalStrings(got, want) {
		t.Errorf("after Filter, PartialType.Methods = %v, want %v", got, want)
	}
}
