// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const parDiscardSrc = `package p

type closer struct{}

func (*closer) Close() error { return nil }

type valueCloser struct{}

func (valueCloser) Close() error { return nil }

type embedsCloser struct{ *closer }

type nearMiss struct{}

func (*nearMiss) Close(force bool) error { return nil }

type failure struct{}

func (*failure) Error() string { return "" }

type wrongResult struct{}

func (*wrongResult) Close() *failure { return nil }

type plain struct{}

type shape interface{ Area() int }

func generic[T any](T) {}
`

func checkParDiscardPackage(t *testing.T) *types2.Package {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase("discard.go"), strings.NewReader(parDiscardSrc), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	conf := types2.Config{}
	pkg, err := conf.Check("p", []*syntax.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}

func TestParDiscardFollowsTheResultType(t *testing.T) {
	pkg := checkParDiscardPackage(t)
	named := func(name string) types2.Type { return pkg.Scope().Lookup(name).Type() }
	pointer := func(name string) types2.Type { return types2.NewPointer(named(name)) }
	typeParam := named("generic").(*types2.Signature).TypeParams().At(0)
	cases := []struct {
		name    string
		typ     types2.Type
		spelled string
		want    string
	}{
		{"pointer to a type with a pointer-receiver Close", pointer("closer"), "*closer", "CloseResult"},
		{"value of a type with a pointer-receiver Close", named("closer"), "closer", "LeaveResultOpen"},
		{"value-receiver Close", named("valueCloser"), "valueCloser", "CloseResult"},
		{"Close promoted through an embedded pointer", named("embedsCloser"), "embedsCloser", "CloseResult"},
		{"Close with a parameter", pointer("nearMiss"), "*nearMiss", "LeaveResultOpen"},
		{"Close with a result other than error", pointer("wrongResult"), "*wrongResult", "LeaveResultOpen"},
		{"type without Close", pointer("plain"), "*plain", "LeaveResultOpen"},
		{"basic type", types2.Typ[types2.Int], "int", "LeaveResultOpen"},
		{"slice type", types2.NewSlice(types2.Typ[types2.Int]), "[]int", "LeaveResultOpen"},
		{"named interface", named("shape"), "shape", "CloseResultIfCloser"},
		{"empty interface", types2.NewInterfaceType(nil, nil), "interface{}", "CloseResultIfCloser"},
		{"type parameter", typeParam, "T", "CloseResultIfCloser"},
		{"closer spelled as any", pointer("closer"), "any", "CloseResultIfCloser"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := parDiscard(c.typ, c.spelled); got != c.want {
				t.Fatalf("parDiscard(%s, %q) = %s, want %s", c.typ, c.spelled, got, c.want)
			}
		})
	}
}
