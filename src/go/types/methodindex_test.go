// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"fmt"
	"go/types"
	"strings"
	"testing"
)

func TestImplementsLargeEmbeddedGeneric(t *testing.T) {
	const n = 96
	var src strings.Builder
	fmt.Fprint(&src, "package p\ntype Base[T any] struct{}\ntype Outer struct{ *Base[int] }\n")
	for i := n - 1; i >= 0; i-- {
		fmt.Fprintf(&src, "func (*Base[T]) Method%02d(T) {}\n", i)
	}
	for _, constraint := range []string{"int", "string"} {
		fmt.Fprintf(&src, "type Interface%s interface {\n", constraint)
		for i := 0; i < n; i++ {
			param := "int"
			if i == n-1 {
				param = constraint
			}
			fmt.Fprintf(&src, " Method%02d(%s)\n", i, param)
		}
		fmt.Fprintln(&src, "}")
	}
	pkg := mustTypecheck(src.String(), nil, nil)
	outer := pkg.Scope().Lookup("Outer").Type()
	getInterface := func(name string) *types.Interface {
		return pkg.Scope().Lookup(name).Type().Underlying().(*types.Interface)
	}

	if good := getInterface("Interfaceint"); !types.Implements(outer, good) {
		t.Fatalf("%s does not implement %s", outer, good)
	}
	bad := getInterface("Interfacestring")
	m, wrongType := types.MissingMethod(outer, bad, true)
	want := fmt.Sprintf("Method%02d", n-1)
	if m == nil || m.Name() != want || !wrongType {
		t.Fatalf("MissingMethod returned (%v, %t), want (%s, true)", m, wrongType, want)
	}
}
