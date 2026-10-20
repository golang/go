// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"reflect"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

const chainPosSource = `package p

func f(ch chan int, x any) {
	{
		_ = []int{1, 2}
	}
	switch x {
	case 1:
	default:
	}
	select {
	case <-ch:
	default:
	}
	_ = func() {}
}
`

// TestNormalizeChainPos checks that normalizing a fragment moves every
// node position and every raw delimiter position onto the base, so no
// position of a chain fragment survives into scope-order checks.
func TestNormalizeChainPos(t *testing.T) {
	file, err := syntax.Parse(syntax.NewFileBase("frag.go"), strings.NewReader(chainPosSource), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	fn := file.DeclList[0].(*syntax.FuncDecl)
	want := syntax.MakePos(syntax.NewFileBase("logos-chain"), 1, 1)
	normalizeChainPos(fn, want)
	posType := reflect.TypeFor[syntax.Pos]()
	syntax.Inspect(fn, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		if n.Pos() != want {
			t.Errorf("%T: node position %v, want %v", n, n.Pos(), want)
		}
		v := reflect.ValueOf(n).Elem()
		for i := range v.NumField() {
			f := v.Type().Field(i)
			if f.Type != posType || !f.IsExported() {
				continue
			}
			if got := v.Field(i).Interface().(syntax.Pos); got != want {
				t.Errorf("%T.%s = %v, want %v", n, f.Name, got, want)
			}
		}
		return true
	})
}
