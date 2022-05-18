// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package govulncheck

import (
	"strings"
	"testing"

	"golang.org/x/vuln/vulncheck"
)

func TestPkgPath(t *testing.T) {
	for _, test := range []struct {
		in   vulncheck.FuncNode
		want string
	}{
		{
			vulncheck.FuncNode{PkgPath: "math", Name: "Floor"},
			"math",
		},
		{
			vulncheck.FuncNode{RecvType: "a.com/b.T", Name: "M"},
			"a.com/b",
		},
		{
			vulncheck.FuncNode{RecvType: "*a.com/b.T", Name: "M"},
			"a.com/b",
		},
	} {
		got := PkgPath(&test.in)
		if got != test.want {
			t.Errorf("%+v: got %q, want %q", test.in, got, test.want)
		}
	}
}

func TestSummarizeCallStack(t *testing.T) {
	topPkgs := map[string]bool{"t1": true, "t2": true}
	vulnPkg := "v"

	for _, test := range []struct {
		in, want string
	}{
		{"a.F", ""},
		{"t1.F", ""},
		{"v.V", ""},
		{
			"t1.F v.V",
			"t1.F calls v.V",
		},
		{
			"t1.F t2.G v.V1 v.v2",
			"t2.G calls v.V1",
		},
		{
			"t1.F x.Y t2.G a.H b.I c.J v.V",
			"t2.G calls a.H, which eventually calls v.V",
		},
	} {
		in := stringToCallStack(test.in)
		got := SummarizeCallStack(in, topPkgs, vulnPkg)
		if got != test.want {
			t.Errorf("%s:\ngot  %s\nwant %s", test.in, got, test.want)
		}
	}
}

func stringToCallStack(s string) vulncheck.CallStack {
	var cs vulncheck.CallStack
	for _, e := range strings.Fields(s) {
		parts := strings.Split(e, ".")
		cs = append(cs, vulncheck.StackEntry{
			Function: &vulncheck.FuncNode{
				PkgPath: parts[0],
				Name:    parts[1],
			},
		})
	}
	return cs
}
