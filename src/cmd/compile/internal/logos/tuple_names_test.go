// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestTupleTemporaryNamesAcrossFiles(t *testing.T) {
	const a = `package p

var A = pair(11)
var A0, A1 = A
`
	const b = `package p

var B = pair(22)
var B0, B1 = B
`
	const funcs = `package p

func pair(n int) (int, int) { return n, n + 1 }
func unpack() (int, int) { return A }
func local() int {
	x := pair(33)
	y := pair(44)
	x0, x1 := x
	y0, y1 := y
	var z = pair(55)
	var z0, z1 = z
	return x0 + x1 + y0 + y1 + z0 + z1
}
`
	for _, tc := range []struct {
		name string
		srcs []string
	}{
		{"forward", []string{a, b, funcs}},
		{"reverse", []string{b, a, funcs}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			first := lowerTupleNamePackage(t, tc.srcs)
			second := lowerTupleNamePackage(t, tc.srcs)
			if !slices.Equal(first, second) {
				t.Fatal("lowering the same package twice changed its generated source")
			}
		})
	}
}

func lowerTupleNamePackage(t *testing.T, sources []string) []string {
	t.Helper()
	var files []*syntax.File
	src := make(map[*syntax.File][]byte)
	for i, source := range sources {
		name := string(rune('a'+i)) + ".lgo"
		f, err := syntax.Parse(syntax.NewFileBase(name), strings.NewReader(source), nil, nil, syntax.CheckBranches|syntax.LogosMode)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
		src[f] = []byte(source)
	}
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	conf := types2.Config{Logos: true}
	pkg, err := conf.Check("p", files, info)
	if err != nil {
		t.Fatalf("source check: %v", err)
	}
	changed, edits := Lower(pkg, info, files, src)
	if !changed {
		t.Fatal("tuple lowering made no changes")
	}
	var output []string
	var ejected []*syntax.File
	for _, f := range files {
		data, err := ApplyEdits(src[f], f.Pos().FileBase(), edits[f])
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, string(data))
		g, err := syntax.Parse(syntax.NewFileBase(f.Pos().FileBase().Filename()), strings.NewReader(string(data)), nil, nil, syntax.CheckBranches)
		if err != nil {
			t.Fatalf("ejected parse: %v\n%s", err, data)
		}
		ejected = append(ejected, g)
	}
	for _, tc := range []struct {
		name  string
		files []*syntax.File
	}{
		{"native", files},
		{"ejected", ejected},
	} {
		var errs []error
		plain := types2.Config{Error: func(err error) { errs = append(errs, err) }}
		_, _ = plain.Check("p", tc.files, nil)
		if err := errors.Join(errs...); err != nil {
			t.Errorf("%s tuple package does not recheck: %v", tc.name, err)
		}
	}
	return output
}
