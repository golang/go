// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package typeutil_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"sort"

	"golang.org/x/tools/go/types/typeutil"
)

func ExampleMap() {
	const source = `package P

var X []string
var Y []string

const p, q = 1.0, 2.0

func f(offset int32) (value byte, ok bool)
func g(rune) (uint8, bool)
`

	// Parse and type-check the package.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "P.go", source, 0)
	if err != nil {
		panic(err)
	}
	pkg, err := new(types.Config).Check("P", fset, []*ast.File{f}, nil)
	if err != nil {
		panic(err)
	}

	scope := pkg.Scope()

	// Group names of package-level objects by their type.
	var namesByType typeutil.Map // value is []string
	for _, name := range scope.Names() {
		T := scope.Lookup(name).Type()

		names, _ := namesByType.At(T).([]string)
		names = append(names, name)
		namesByType.Set(T, names)
	}

	// Format, sort, and print the map entries.
	var lines []string
	namesByType.Iterate(func(T types.Type, names interface{}) {
		lines = append(lines, fmt.Sprintf("%s   %s", names, T))
	})
	sort.Strings(lines)
	for _, line := range lines {
		fmt.Println(line)
	}

	// Output:
	// [X Y]   []string
	// [f g]   func(offset int32) (value byte, ok bool)
	// [p q]   untyped float
}
