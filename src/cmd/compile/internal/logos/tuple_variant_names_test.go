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

type tupleNamedSource struct {
	name string
	text string
}

const tupleVariantCommon = "package p\nfunc pair(n int) (int, int) { return n, n + 1 }\n"
const tupleVariantA = "package p\n\nvar A = pair(11)\nvar A0, A1 = A\n"
const tupleVariantB = "package p\n\n\n\nvar B = pair(22)\nvar B0, B1 = B\n"

func TestTupleTemporaryNamesComposeVariants(t *testing.T) {
	for _, names := range [][2]string{{"a.lgo", "b.lgo"}, {"a-b.lgo", "a_b.lgo"}} {
		t.Run(names[0]+"+"+names[1], func(t *testing.T) {
			common := tupleNamedSource{"common.lgo", tupleVariantCommon}
			a := lowerTupleNamedSources(t, []tupleNamedSource{common, {names[0], tupleVariantA}})
			b := lowerTupleNamedSources(t, []tupleNamedSource{common, {names[1], tupleVariantB}})
			for form, label := range []string{"native", "ejected"} {
				// Inactive-variant ejection retains each optional file from
				// a separate compiler invocation, then allows both tags.
				checkTupleNamedGo(t, label, []tupleNamedSource{
					common, {names[0], a[1][form]}, {names[1], b[1][form]},
				})
			}
		})
	}
}

func TestTupleTemporaryNamesDuplicateBasenames(t *testing.T) {
	// Lower also accepts direct compiler input spanning directories, even
	// though cmd/go packages take their files from one directory.
	sources := []tupleNamedSource{
		{"common.lgo", tupleVariantCommon},
		{"left/variant.lgo", tupleVariantA},
		{"right/variant.lgo", tupleVariantB},
	}
	outputs := lowerTupleNamedSources(t, sources)
	for form, label := range []string{"native", "ejected"} {
		var lowered []tupleNamedSource
		for i, source := range sources {
			lowered = append(lowered, tupleNamedSource{source.name, outputs[i][form]})
		}
		checkTupleNamedGo(t, label, lowered)
	}
}

func lowerTupleNamedSources(t *testing.T, sources []tupleNamedSource) [][2]string {
	t.Helper()
	var files []*syntax.File
	src := make(map[*syntax.File][]byte)
	for _, source := range sources {
		f, err := syntax.Parse(syntax.NewFileBase(source.name), strings.NewReader(source.text), nil, nil, syntax.CheckBranches|syntax.LogosMode)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
		src[f] = []byte(source.text)
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
	_, edits := Lower(pkg, info, files, src)
	var output [][2]string
	for _, file := range files {
		var native strings.Builder
		if _, err := syntax.Fprint(&native, file, 0); err != nil {
			t.Fatal(err)
		}
		ejected, err := ApplyEdits(src[file], file.Pos().FileBase(), edits[file])
		if err != nil {
			t.Fatal(err)
		}
		output = append(output, [2]string{native.String(), string(ejected)})
	}
	return output
}

func checkTupleNamedGo(t *testing.T, form string, sources []tupleNamedSource) {
	t.Helper()
	var files []*syntax.File
	for _, source := range sources {
		f, err := syntax.Parse(syntax.NewFileBase(source.name), strings.NewReader(source.text), nil, nil, syntax.CheckBranches)
		if err != nil {
			t.Fatalf("%s parse: %v", form, err)
		}
		files = append(files, f)
	}
	conf := types2.Config{}
	if _, err := conf.Check("p", files, nil); err != nil {
		t.Errorf("%s combined tuple files do not recheck: %v", form, err)
	}
}
