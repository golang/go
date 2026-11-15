// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts_test

import (
	"errors"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func checkDupSources(t *testing.T, sources map[string]string) (*types2.Package, *types2.Info, []*syntax.File, map[*syntax.File][]byte) {
	t.Helper()
	dir := t.TempDir()
	var files []*syntax.File
	src := make(map[*syntax.File][]byte)
	for _, name := range slices.Sorted(maps.Keys(sources)) {
		body := sources[name]
		path := filepath.Join(dir, name)
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
		var perrs []error
		file, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.LogosMode)
		if err != nil {
			t.Fatal(errors.Join(append(perrs, err)...))
		}
		files = append(files, file)
		src[file] = []byte(body)
	}
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	var checkErrs []error
	conf := types2.Config{
		Error: func(err error) { checkErrs = append(checkErrs, err) },
		Logos: true,
	}
	pkg, _ := conf.Check("duptest", files, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	return pkg, info, files, src
}

func TestWriterEmitsNoDuplicateKeys(t *testing.T) {
	fileA := `package duptest

type Pair struct {
	Name string ` + "`json:\"name\"`" + `
	Qty  int    ` + "`json:\"qty,omitempty\"`" + `
}

func (Pair) UnmarshalJSON(b []byte) error { return nil }

func Dup(v []byte) { v[0] = 1 }

func ProbeOpt(v int?) int { return v ?? 0 }

func Sum(xs const []int) int {
	total := 0
	for _, x := range xs {
		total += x
	}
	return total
}

func Gen[T any](v T) T { return v }

var _ = Gen[int](1)
var _ = Gen[string]("s")

const func Twice(n int) int { return n + n }
`
	fileB := `package duptest

type DupBox struct{ N int }

func (DupBox) Dup(v []byte) { v[0] = 1 }

type Vec struct {
	X int
	Y int
}

type Carrier interface {
	Pos Vec
}

type BotA struct : Carrier {
	Pos Vec
}

type BotB struct : Carrier {
	Pos Vec
}
`
	pkg, info, files, src := checkDupSources(t, map[string]string{"a.lgo": fileA, "b.lgo": fileB})
	got := facts.Write(pkg, info, files, src)
	fnCounts := make(map[string]int)
	for _, fn := range got.Fns {
		fnCounts[fn.Object]++
	}
	for obj, n := range fnCounts {
		if n > 1 {
			t.Errorf("writer emitted %d fn records for %q", n, obj)
		}
	}
	if fnCounts["Dup"] != 1 {
		t.Errorf("fn records for Dup = %d, want 1", fnCounts["Dup"])
	}
	if fnCounts["DupBox.Dup"] != 1 {
		t.Errorf("fn records for DupBox.Dup = %d, want 1", fnCounts["DupBox.Dup"])
	}
	for _, c := range got.Comptimes {
		t.Errorf("source-phase Write emitted comptime record for %q", c.Object)
	}
	for _, c := range got.ConstFuncs {
		if c.Object == "Gen" {
			t.Errorf("writer emitted constfunc record for generic Gen")
		}
	}
	foundTwice := false
	for _, c := range got.ConstFuncs {
		if c.Object == "Twice" {
			foundTwice = true
		}
	}
	if !foundTwice {
		t.Errorf("missing constfunc record for Twice (have %+v)", got.ConstFuncs)
	}
	if len(got.Conforms) != 2 {
		t.Errorf("conforms records = %+v, want one per asserting struct", got.Conforms)
	}
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	if _, err := facts.Decode(facts.Encode(fp, got)); err != nil {
		t.Fatalf("Decode(Encode(writer output)) failed: %v", err)
	}
}
