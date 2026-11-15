// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"go/constant"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestWriteConstFnFactsAliasClosure(t *testing.T) {
	const src = `package constfntest

type tally int

type Tally = tally

type Val struct {
	weight int
	note   string
	tally  Tally
}

func (v Val) Weight() int {
	return v.weight
}

func (v Val) Doubled() int {
	return v.weight * 2
}

func (v Val) Note() string {
	return v.note
}

func (v Val) Tally() int {
	return int(v.tally)
}

const func MakeVal(weight int, note string, tally int) Val {
	return Val{weight: weight, note: note, tally: Tally(tally)}
}

const func WeightOf(v Val) int {
	return v.weight
}

const func DoubledOf(v Val) int {
	return v.weight * 2
}

const Seed = Tally(7)

const func Seeded() Val {
	return Val{weight: 1, note: "seed", tally: Seed}
}

const tallyOf = Tally(7)
const seededVal = Seeded()
`
	pkg, info, file := checkConstFnFixture(t, src)
	srcBytes := []byte(src)
	f := facts.Write(pkg, info, []*syntax.File{file}, map[*syntax.File][]byte{file: srcBytes})
	probe := map[string]string{
		"MakeVal":   "const tallyOf = Tally(7)\nconst made = MakeVal(3, \"n\", 4)",
		"WeightOf":  "const tallyOf = Tally(7)\nconst v3 = Val{weight: 3, note: \"n\", tally: 4}\nconst vWeight = WeightOf(v3)",
		"DoubledOf": "const tallyOf = Tally(7)\nconst v3 = Val{weight: 3, note: \"n\", tally: 4}\nconst vDouble = DoubledOf(v3)",
		"Seeded":    "const tallyOf = Tally(7)\nconst seeded = Seeded()",
	}
	seen := map[string]bool{}
	for _, rec := range f.ConstFuncs {
		p, ok := probe[rec.Object]
		if !ok {
			t.Fatalf("unexpected constfn record %q", rec.Object)
		}
		if seen[rec.Object] {
			t.Fatalf("duplicate constfn record %q", rec.Object)
		}
		seen[rec.Object] = true
		_, scope := checkConstFnPayload(t, rec, p)
		tn, ok := scope.Lookup("Tally").(*types2.TypeName)
		if !ok || !tn.IsAlias() {
			t.Errorf("constfn payload for %s: Tally lost alias identity", rec.Object)
		}
		if scope.Lookup("tally") == nil {
			t.Errorf("constfn payload for %s: missing alias target tally", rec.Object)
		}
		if scope.Lookup("Val") == nil {
			t.Errorf("constfn payload for %s: missing Val", rec.Object)
		}
		wantScopeConstAlias(t, scope, rec.Object, "tallyOf", "tally", 7)
		switch rec.Object {
		case "MakeVal":
			wantScopeStructAlias(t, scope, rec.Object, "made", []string{"int", "string"}, "tally")
		case "WeightOf":
			wantScopeStructAlias(t, scope, rec.Object, "v3", []string{"int", "string"}, "tally")
			wantScopeConst(t, scope, rec.Object, "vWeight", "int", 3)
		case "DoubledOf":
			wantScopeStructAlias(t, scope, rec.Object, "v3", []string{"int", "string"}, "tally")
			wantScopeConst(t, scope, rec.Object, "vDouble", "int", 6)
		case "Seeded":
			wantScopeConstAlias(t, scope, rec.Object, "Seed", "tally", 7)
			wantScopeStructAlias(t, scope, rec.Object, "seeded", []string{"int", "string"}, "tally")
		}
	}
	for name := range probe {
		if !seen[name] {
			t.Errorf("missing constfn record %q", name)
		}
	}
}

func checkConstFnPayload(t *testing.T, rec facts.ConstFuncFact, probe string) (string, *types2.Scope) {
	t.Helper()
	text := parseConstFnPayload(t, rec) + "\n" + probe + "\n"
	dir := t.TempDir()
	path := filepath.Join(dir, "constfn.lgo")
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("reparse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	imp := newTestImporter(t)
	var checkErrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { checkErrs = append(checkErrs, err) },
		Logos:    true,
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	pkg, _ := conf.Check("constfntest", []*syntax.File{file}, info)
	for _, err := range checkErrs {
		t.Errorf("recheck %s: %v", rec.Object, err)
	}
	if len(checkErrs) > 0 {
		t.FailNow()
	}
	return text, pkg.Scope()
}

func wantScopeConstAlias(t *testing.T, scope *types2.Scope, rec, name, target string, want int64) {
	t.Helper()
	obj := scope.Lookup(name)
	c, ok := obj.(*types2.Const)
	if !ok {
		t.Fatalf("constfn payload for %s: %s = %T, want a constant", rec, name, obj)
	}
	tn, ok := scope.Lookup(target).(*types2.TypeName)
	if !ok || tn == nil {
		t.Fatalf("constfn payload for %s: missing target type %s", rec, target)
	}
	got := types2.Unalias(c.Type())
	if got != tn.Type() && !types2.Identical(got, tn.Type()) {
		t.Errorf("constfn payload for %s: %s has type %s, want exactly the payload's own %s", rec, name, c.Type(), target)
	}
	v, ok := constant.Int64Val(c.Val())
	if !ok || v != want {
		t.Errorf("constfn payload for %s: %s = %v, want %d", rec, name, c.Val(), want)
	}
}

func wantScopeStructAlias(t *testing.T, scope *types2.Scope, rec, name string, plain []string, target string) {
	t.Helper()
	obj := scope.Lookup(name)
	if obj == nil {
		t.Fatalf("constfn payload for %s: missing %s", rec, name)
	}
	st, ok := obj.Type().Underlying().(*types2.Struct)
	if !ok {
		t.Fatalf("constfn payload for %s: %s has type %v, want a struct", rec, name, obj.Type())
	}
	if got := st.NumFields(); got != len(plain)+1 {
		t.Fatalf("constfn payload for %s: %s has %d fields, want %d", rec, name, got, len(plain)+1)
	}
	tn, ok := scope.Lookup(target).(*types2.TypeName)
	if !ok || tn == nil {
		t.Fatalf("constfn payload for %s: missing target type %s", rec, target)
	}
	for i, wt := range plain {
		if got := st.Field(i).Type().String(); got != wt {
			t.Errorf("constfn payload for %s: %s field %d has type %s, want %s", rec, name, i, got, wt)
		}
	}
	aliasField := len(plain)
	got := types2.Unalias(st.Field(aliasField).Type())
	if got != tn.Type() && !types2.Identical(got, tn.Type()) {
		t.Errorf("constfn payload for %s: %s field %d has type %s, want exactly the payload's own %s", rec, name, aliasField, st.Field(aliasField).Type(), target)
	}
}
