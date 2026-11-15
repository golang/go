// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestHelperParseSeverityRoundTrip(t *testing.T) {
	for name, want := range map[string]Severity{
		"forbid": Forbid,
		"deny":   Deny,
		"warn":   Warn,
		"allow":  Allow,
	} {
		got, ok := ParseSeverity(name)
		if !ok || got != want {
			t.Errorf("ParseSeverity(%q) = %v, %v; want %v, true", name, got, ok, want)
		}
		if got.String() != name {
			t.Errorf("Severity(%v).String() = %q, want %q", got, got.String(), name)
		}
	}
	for _, name := range []string{"", "loud", "FORBID", "error"} {
		if got, ok := ParseSeverity(name); ok || got != Allow {
			t.Errorf("ParseSeverity(%q) = %v, %v; want Allow, false", name, got, ok)
		}
	}
}

func TestHelperIsTestFileNames(t *testing.T) {
	for name, want := range map[string]bool{
		"x_test.go":   true,
		"x_test.lgo":  true,
		"x_test.slog": true,
		"x.go":        false,
		"x.lgo":       false,
		"xtest.go":    false,
	} {
		file, _ := syntax.Parse(syntax.NewFileBase(name), strings.NewReader("package p\n"), nil, nil, 0)
		if file == nil {
			t.Fatalf("%s did not parse", name)
		}
		if got := IsTestFile(file); got != want {
			t.Errorf("IsTestFile(%s) = %v, want %v", name, got, want)
		}
	}
}

func TestHelperDeclNameForms(t *testing.T) {
	file, _ := syntax.Parse(syntax.NewFileBase("x.go"), strings.NewReader("package p\n\nfunc F() {}\n\ntype T struct{}\n\nfunc (t T) M() {}\n\nfunc (t *T) N() {}\n"), nil, nil, 0)
	if file == nil {
		t.Fatal("decls did not parse")
	}
	var got []string
	for _, d := range file.DeclList {
		if fn, ok := d.(*syntax.FuncDecl); ok {
			got = append(got, DeclName(fn))
		}
	}
	want := []string{"F", "T.M", "(*T).N"}
	if len(got) != len(want) {
		t.Fatalf("DeclName = %q, want %q", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("DeclName[%d] = %q, want %q", i, got[i], want[i])
		}
	}
}

func TestHelperCallEndNeedsClosingParen(t *testing.T) {
	const src = "package p\n\nfunc f() {\n  g(1)\n}\n\nfunc g(x int) {}\n"
	in := check(t, KindGo, "x.go", src)
	var call *syntax.CallExpr
	syntax.Inspect(in.Files[0], func(n syntax.Node) bool {
		if c, ok := n.(*syntax.CallExpr); ok && call == nil {
			call = c
		}
		return true
	})
	if call == nil {
		t.Fatal("no call found")
	}
	plain := &Pass{Files: in.Files, Kind: in.Kind, in: &Input{}}
	if end := plain.CallEnd(in.Files[0], call); end.Line() != 4 || end.Col() != 6 {
		t.Errorf("sourceless end = %d:%d, want 4:6", end.Line(), end.Col())
	}
	full := &Pass{Files: in.Files, Kind: in.Kind, in: in}
	if end := full.CallEnd(in.Files[0], call); end.Line() != 4 || end.Col() != 7 {
		t.Errorf("sourced end = %d:%d, want 4:7", end.Line(), end.Col())
	}
}

func TestHelperFallbackActiveFollowsPolicy(t *testing.T) {
	if !FallbackActive(&Input{Kind: KindLgo}) || !FallbackActive(&Input{Kind: KindGo}) {
		t.Fatal("fallback must be active under the default policy")
	}
	p, err := ParsePolicy([]byte(`{"version":1,"groups":{"perf":"allow"}}`))
	if err != nil {
		t.Fatal(err)
	}
	if FallbackActive(&Input{Kind: KindLgo, Policy: p}) {
		t.Error("fallback must be idle when perf is allowed")
	}
}

func TestHelperLockFactsRequires(t *testing.T) {
	const src = "package p\n\nimport \"sync\"\n\n//logos:requires mu\nfunc f(mu *sync.Mutex) {\n\tmu.Lock()\n\tdefer mu.Unlock()\n}\n"
	in := check(t, KindGo, "x.go", src)
	got := LockFacts(in)
	if len(got) != 1 {
		t.Fatalf("LockFacts = %+v, want one contract", got)
	}
	if got[0].Effect != "requires" || got[0].Path != "mu" {
		t.Errorf("LockFacts = %+v, want requires mu", got)
	}
	if !strings.Contains(got[0].Object, "f") {
		t.Errorf("LockFacts object = %q, want the function", got[0].Object)
	}
	if len(LockFacts(&Input{})) != 0 {
		t.Error("LockFacts of a package without contracts must be empty")
	}
}

func TestHelperTypeOfRecorded(t *testing.T) {
	if TypeOf(nil) != nil {
		t.Error("TypeOf(nil) must be nil")
	}
	mode := syntax.CheckBranches
	file, _ := syntax.Parse(syntax.NewFileBase("x.go"), strings.NewReader("package p\nvar v = 1\n"), nil, nil, mode)
	if file == nil {
		t.Fatal("source did not parse")
	}
	info := &types2.Info{
		StoreTypesInSyntax: true,
		Defs:               make(map[*syntax.Name]types2.Object),
		Uses:               make(map[*syntax.Name]types2.Object),
	}
	conf := &types2.Config{Importer: &lintImporter{pkgs: make(map[string]*types2.Package)}}
	if _, err := conf.Check("p", []*syntax.File{file}, info); err != nil {
		t.Fatal(err)
	}
	var lit syntax.Expr
	syntax.Inspect(file, func(n syntax.Node) bool {
		if b, ok := n.(*syntax.BasicLit); ok && lit == nil {
			lit = b
		}
		return true
	})
	if lit == nil {
		t.Fatal("no literal found")
	}
	b, ok := TypeOf(lit).(*types2.Basic)
	if !ok || b.Kind() != types2.Int {
		t.Errorf("TypeOf(1) = %v, want basic int", TypeOf(lit))
	}
}
