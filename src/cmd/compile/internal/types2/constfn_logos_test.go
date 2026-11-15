// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"strings"
	"testing"
)

func logosTestChecker() *Checker {
	return &Checker{conf: &Config{Logos: true}, Info: &Info{}}
}

func logosTestUnit(t *testing.T, src string) *logosConstFnUnit {
	t.Helper()
	_, _, u, err := logosParseConstFnUnit("example.com/u", "F", []byte(src))
	if err != nil {
		t.Fatalf("logosParseConstFnUnit: %v", err)
	}
	u.defPkg = NewPackage("example.com/u", "u")
	return u
}

const logosTestRoot = "\nconst func F() int { return 0 }\n"

func TestLogosParseConstFnUnitValidation(t *testing.T) {
	valid := `package p

const base = 10

const func F(x int) int {
	return x + base
}

func g(y int) int {
	return y
}
`
	tests := []struct {
		name   string
		src    string
		object string
		want   string
	}{
		{"valid", valid, "F", ""},
		{"missing root", valid, "Z", "has no const func Z"},
		{"plain same name", "package p\nfunc F(x int) int { return x }\n", "F", "declares plain function F"},
		{"method root", "package p\ntype T int\nconst func (t T) F() int { return 1 }\n", "F", "declares method F"},
		{"generic root", "package p\nconst func F[T any](x T) T { return x }\n", "F", "declares generic function F"},
		{"no body", "package p\nconst func F(x int) int\n", "F", "has no body"},
		{"parse error", "package p\nconst func F(\n", "F", "cannot parse const record"},
		{"duplicate", "package p\nconst func F() int { return 1 }\nfunc g() int { return 1 }\nfunc g() int { return 2 }\n", "F", "declares g twice"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, root, u, err := logosParseConstFnUnit("example.com/u", tt.object, []byte(tt.src))
			if tt.want == "" {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if root != u.root || root.Name.Value != tt.object {
					t.Fatalf("root = %v, want %s", root, tt.object)
				}
				return
			}
			if err == nil {
				t.Fatalf("got nil error, want %q", tt.want)
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("error %q does not contain %q", err, tt.want)
			}
		})
	}
}

func TestLogosIndexUnitConstsGroups(t *testing.T) {
	u := logosTestUnit(t, `package p

const Standalone = 7

const (
	A = iota
	B
	C = iota + 10
)

const (
	Solo int = 3
)
`+logosTestRoot)
	if got := u.specs["Standalone"]; got == nil || got.iota != 0 {
		t.Fatalf("Standalone ordinal = %v, want 0", got)
	}
	for name, want := range map[string]int{"A": 0, "B": 1, "C": 2, "Solo": 0} {
		spec, ok := u.specs[name]
		if !ok {
			t.Fatalf("missing spec %s", name)
		}
		if spec.iota != want {
			t.Fatalf("spec %s ordinal = %d, want %d", name, spec.iota, want)
		}
	}
}

func TestLogosUnitConstValueIota(t *testing.T) {
	check := logosTestChecker()
	check.logosConstFnMaps()
	u := logosTestUnit(t, `package p

const (
	A = iota
	B
	C = iota + 10
)

const K = iota + 1
`+logosTestRoot)
	check.logosConstFnBases[u.base] = u
	for name, want := range map[string]int64{"A": 0, "B": 1, "C": 12, "K": 1} {
		v, err := check.logosUnitConstValue(u, name)
		if err != nil {
			t.Fatalf("const %s: %v", name, err)
		}
		if v.Int64() != want {
			t.Fatalf("const %s = %d, want %d", name, v.Int64(), want)
		}
	}
}

func TestLogosUnitConstValueDeclaredType(t *testing.T) {
	check := logosTestChecker()
	check.logosConstFnMaps()
	u := logosTestUnit(t, `package p

type myInt int64

const (
	W8 int8 = 100
	M  myInt = 5
	U  = 6
)
`+logosTestRoot)
	check.logosConstFnBases[u.base] = u
	v, err := check.logosUnitConstValue(u, "W8")
	if err != nil {
		t.Fatalf("W8: %v", err)
	}
	if v.Type() == nil || v.Type().Kind != ctfe.KInt8 {
		t.Fatalf("W8 type = %+v, want KInt8", v.Type())
	}
	if v.Int64() != 100 {
		t.Fatalf("W8 = %d, want 100", v.Int64())
	}
	m, err := check.logosUnitConstValue(u, "M")
	if err != nil {
		t.Fatalf("M: %v", err)
	}
	if m.Type() == nil || m.Type().Named != "myInt" {
		t.Fatalf("M type = %+v, want Named myInt", m.Type())
	}
	if m.Int64() != 5 {
		t.Fatalf("M = %d, want 5", m.Int64())
	}
	x, err := check.logosUnitConstValue(u, "U")
	if err != nil {
		t.Fatalf("U: %v", err)
	}
	if x.Int64() != 6 {
		t.Fatalf("U = %d, want 6", x.Int64())
	}
}

func TestLogosUnitConstValueFreshValuesStayUntyped(t *testing.T) {
	check := logosTestChecker()
	check.logosConstFnMaps()
	u := logosTestUnit(t, `package p

const (
	A int8 = 1
	B = 256
	C
)
`+logosTestRoot)
	check.logosConstFnBases[u.base] = u
	for _, name := range []string{"B", "C"} {
		v, err := check.logosUnitConstValue(u, name)
		if err != nil {
			t.Fatalf("const %s: %v", name, err)
		}
		if v.Int64() != 256 {
			t.Fatalf("const %s = %d, want 256", name, v.Int64())
		}
		if v.Type() == nil || v.Type().Kind != ctfe.KUntypedInt {
			t.Fatalf("const %s type = %+v, want untyped int", name, v.Type())
		}
	}
}

func TestLogosUnitConstValueCycle(t *testing.T) {
	check := logosTestChecker()
	u := logosTestUnit(t, `package p

const (
	A = B
	B = A
)

const func F() int { return 0 }
`)
	check.logosConstFnMaps()
	check.logosConstFnBases[u.base] = u
	if _, err := check.logosUnitConstValue(u, "A"); err == nil {
		t.Fatal("cyclic const evaluated without error")
	} else if !strings.Contains(err.Error(), "cycle") && !strings.Contains(err.Error(), "A") {
		t.Fatalf("cycle error %q names neither cycle nor const", err)
	}
}

func TestLogosBuildFuncAnonymousParam(t *testing.T) {
	pkg := NewPackage("example.com/caller", "caller")
	check := logosTestChecker()
	check.pkg = pkg
	host := &ctfeHost{check: check}
	file, err := syntax.Parse(syntax.NewFileBase("anon"), strings.NewReader("package p\nfunc F(int) int { return 1 }\n"), nil, nil, 0)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	fd, ok := file.DeclList[0].(*syntax.FuncDecl)
	if !ok {
		t.Fatalf("decl = %T, want *syntax.FuncDecl", file.DeclList[0])
	}
	sig := NewSignatureType(nil, nil, nil,
		NewTuple(NewVar(nopos, pkg, "", Typ[Int])),
		NewTuple(NewVar(nopos, pkg, "", Typ[Int])), false)
	obj := NewFunc(nopos, pkg, "F", sig)
	pkg.Scope().Insert(obj)
	bodies, consts := check.logosConstFuncBodies()
	bodies[obj.name] = fd
	consts[fd] = true
	call := func(args ...syntax.Expr) *syntax.CallExpr {
		return &syntax.CallExpr{Fun: syntax.NewName(syntax.Pos{}, "F"), ArgList: args}
	}
	lit := func(s string) syntax.Expr {
		return &syntax.BasicLit{Value: s, Kind: syntax.IntLit}
	}
	v, _, err := ctfe.Eval(host, ctfe.Config{}, call(lit("7")))
	if err != nil {
		t.Fatalf("Eval F(7): %v", err)
	}
	if v.Int64() != 1 {
		t.Fatalf("Eval F(7) = %d, want 1", v.Int64())
	}
	if _, _, err := ctfe.Eval(host, ctfe.Config{}, call()); err == nil {
		t.Fatal("Eval F() succeeded, want wrong-arity rejection")
	} else if !strings.Contains(err.Error(), "wrong number of arguments") {
		t.Fatalf("Eval F() error = %q, want wrong-arity rejection", err)
	}
}

func TestLogosBuildUnitFuncKinds(t *testing.T) {
	check := logosTestChecker()
	host := &ctfeHost{check: check}
	u := logosTestUnit(t, `package p

const func F() int { return 1 }

func plain(x int) int { return x }

func anon(int) int { return 0 }

const func F2[T any](x T) T { return x }
`)
	if fn := host.buildUnitConstFunc(u, u.funcs["F"]); fn == nil || !fn.Const || fn.Body == nil {
		t.Fatalf("const helper = %+v, want Const body", fn)
	}
	if fn := host.buildUnitPlainFunc(u, u.funcs["plain"]); fn == nil || fn.Const || fn.Body == nil || len(fn.Params) != 1 {
		t.Fatalf("plain helper = %+v, want one-param non-const body", fn)
	}
	if fn := host.buildUnitPlainFunc(u, u.funcs["anon"]); fn == nil || len(fn.Params) != 1 {
		t.Fatalf("anonymous-param helper = %+v, want arity-preserving params", fn)
	}
	if fn := host.buildUnitConstFunc(u, u.funcs["F2"]); fn == nil || !fn.Generic {
		t.Fatalf("generic helper = %+v, want Generic", fn)
	}
}

func TestLogosResolveOrdinaryImport(t *testing.T) {
	check := logosTestChecker()
	u, fn, body, err := check.logosResolveImportedConstFunc("example.com/nonexistent", "F")
	if err != nil || u != nil || fn != nil || body != nil {
		t.Fatalf("ordinary import = (%v, %v, %v, %v), want all nil", u, fn, body, err)
	}
	if admitted, detail := check.logosAdmitImportedConstFunc("example.com/nonexistent", "F"); admitted || detail != "" {
		t.Fatalf("ordinary predicate = (%v, %q), want (false, \"\")", admitted, detail)
	}
}

func TestLogosUnitForNode(t *testing.T) {
	check := logosTestChecker()
	check.logosConstFnMaps()
	u := logosTestUnit(t, "package p\nconst func F() int { return 1 }\n")
	check.logosConstFnBases[u.base] = u
	var id *syntax.Name
	syntax.Inspect(u.file, func(n syntax.Node) bool {
		if name, ok := n.(*syntax.Name); ok && name.Value == "F" && id == nil {
			id = name
		}
		return true
	})
	if id == nil {
		t.Fatal("no F name in parsed unit")
	}
	if got := check.logosUnitForNode(id); got != u {
		t.Fatalf("unitForNode(unit name) = %v, want unit", got)
	}
	other := syntax.NewName(syntax.MakePos(syntax.NewFileBase("caller"), 1, 1), "F")
	if got := check.logosUnitForNode(other); got != nil {
		t.Fatalf("unitForNode(caller name) = %v, want nil", got)
	}
	if got := check.logosUnitForNode(nil); got != nil {
		t.Fatalf("unitForNode(nil) = %v, want nil", got)
	}
}
