// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"strings"
	"testing"
)

type logosOwnedTestImporter struct {
	pkgs map[string]*Package
}

func (m *logosOwnedTestImporter) Import(path string) (*Package, error) {
	if p, ok := m.pkgs[path]; ok {
		return p, nil
	}
	return nil, fmt.Errorf("unknown import %q", path)
}

func logosOwnedParseFile(t *testing.T, name, src string) *syntax.File {
	t.Helper()
	f, err := syntax.Parse(syntax.NewFileBase(name), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatalf("syntax.Parse %s: %v", name, err)
	}
	return f
}

func logosOwnedVarValue(t *testing.T, f *syntax.File, i int) syntax.Expr {
	t.Helper()
	n := 0
	for _, decl := range f.DeclList {
		vd, ok := decl.(*syntax.VarDecl)
		if !ok {
			continue
		}
		if n == i {
			if list, ok := vd.Values.(*syntax.ListExpr); ok {
				if len(list.ElemList) != 1 {
					t.Fatalf("var decl %d has %d values", i, len(list.ElemList))
				}
				return list.ElemList[0]
			}
			return vd.Values
		}
		n++
	}
	t.Fatalf("var decl %d not found", i)
	return nil
}

func logosOwnedVarType(t *testing.T, f *syntax.File, i int) syntax.Expr {
	t.Helper()
	n := 0
	for _, decl := range f.DeclList {
		vd, ok := decl.(*syntax.VarDecl)
		if !ok {
			continue
		}
		if n == i {
			return vd.Type
		}
		n++
	}
	t.Fatalf("var decl %d not found", i)
	return nil
}

func logosOwnedChecker(reg map[syntax.Expr]LogosOwnedType, imp *logosOwnedTestImporter, msgs *[]string) *Checker {
	conf := &Config{Importer: imp, LogosOwnedTypes: reg}
	if msgs != nil {
		conf.Error = func(err error) { *msgs = append(*msgs, err.Error()) }
	}
	return NewChecker(conf, NewPackage("example.com/test", "test"), &Info{})
}

func TestLogosOwnedUnregisteredNodeMisses(t *testing.T) {
	f := logosOwnedParseFile(t, "p.go", "package p\nvar _ T\n")
	node := logosOwnedVarType(t, f, 0)
	g := logosOwnedParseFile(t, "q.go", "package p\nvar _ T\n")
	sameSpelling := logosOwnedVarType(t, g, 0)
	if node == sameSpelling {
		t.Fatal("test nodes must be distinct pointers")
	}
	reg := map[syntax.Expr]LogosOwnedType{sameSpelling: LogosOwnedFactory("example.com/fac", "Make",
		LogosOwnedStep{Kind: LogosOwnedSigResult})}
	check := logosOwnedChecker(reg, &logosOwnedTestImporter{}, nil)
	if _, ok := check.logosOwnedTypeOf(node); ok {
		t.Fatal("same spelling on another node must not resolve")
	}
	if check.logosOwnedRegistered(node) {
		t.Fatal("unregistered node must not report registered")
	}
	if !check.logosOwnedRegistered(sameSpelling) {
		t.Fatal("registered node must report registered")
	}

	bare := NewChecker(&Config{}, NewPackage("example.com/test", "test"), &Info{Selections: map[*syntax.SelectorExpr]*Selection{}})
	if _, ok := bare.logosOwnedTypeOf(node); ok {
		t.Fatal("nil map must resolve nothing")
	}
	lit := &syntax.CompositeLit{}
	if _, _, ok := bare.logosOwnedLitField(lit, NewStruct(nil, nil), "secret", nil); ok {
		t.Fatal("nil map must bypass nothing in literals")
	}
	x := &operand{}
	if bare.logosOwnedSelector(x, &syntax.SelectorExpr{Sel: &syntax.Name{Value: "secret"}}) {
		t.Fatal("nil map must bypass nothing in selectors")
	}
}

func logosOwnedSyntheticDEF1(t *testing.T) (defPkg, facPkg *Package, secret *Named, pub *Alias) {
	t.Helper()
	defPkg = NewPackage("example.com/def", "def")
	secretObj := NewTypeName(nopos, defPkg, "secret", nil)
	secret = NewNamed(secretObj, Typ[Int], nil)
	defPkg.Scope().Insert(secretObj)

	facPkg = NewPackage("example.com/fac", "fac")
	mapT := NewMap(Typ[String], secret)
	sig := NewSignatureType(nil, nil, nil, NewTuple(), NewTuple(NewParam(nopos, facPkg, "", mapT)), false)
	facPkg.Scope().Insert(NewFunc(nopos, facPkg, "Make", sig))
	pubObj := NewTypeName(nopos, facPkg, "Pub", nil)
	pub = NewAlias(pubObj, Typ[Int])
	facPkg.Scope().Insert(pubObj)
	pubSig := NewSignatureType(nil, nil, nil, NewTuple(), NewTuple(NewParam(nopos, facPkg, "", pub)), false)
	facPkg.Scope().Insert(NewFunc(nopos, facPkg, "GetPub", pubSig))
	return defPkg, facPkg, secret, pub
}

func TestLogosOwnedFactoryResolvesFresh(t *testing.T) {
	_, fac1, secret1, _ := logosOwnedSyntheticDEF1(t)
	f := logosOwnedParseFile(t, "p.go", "package p\nvar _ T\n")
	node := logosOwnedVarType(t, f, 0)

	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": fac1}}
	reg := map[syntax.Expr]LogosOwnedType{
		node: LogosOwnedFactory("example.com/fac", "Make",
			LogosOwnedStep{Kind: LogosOwnedSigResult},
			LogosOwnedStep{Kind: LogosOwnedMapElem}),
	}
	check1 := logosOwnedChecker(reg, imp, nil)
	got1, ok := check1.logosOwnedTypeOf(node)
	if !ok {
		t.Fatal("registered type node must resolve")
	}
	if got1 != Type(secret1) {
		t.Fatalf("resolution must preserve exact Named identity: got %v, want %v", got1, secret1)
	}

	_, fac2, secret2, _ := logosOwnedSyntheticDEF1(t)
	imp.pkgs["example.com/fac"] = fac2
	check2 := logosOwnedChecker(reg, imp, nil)
	got2, ok := check2.logosOwnedTypeOf(node)
	if !ok {
		t.Fatal("registered type node must resolve after reimport")
	}
	if got2 != Type(secret2) {
		t.Fatalf("resolution must use the fresh import: got %v, want %v", got2, secret2)
	}
	if got2 == Type(secret1) {
		t.Fatal("fresh resolution must not return the stale Named")
	}

	aliasNode := logosOwnedVarType(t, logosOwnedParseFile(t, "a.go", "package p\nvar _ U\n"), 0)
	reg[aliasNode] = LogosOwnedFactory("example.com/fac", "GetPub",
		LogosOwnedStep{Kind: LogosOwnedSigResult})
	check3 := logosOwnedChecker(reg, imp, nil)
	got3, ok := check3.logosOwnedTypeOf(aliasNode)
	if !ok {
		t.Fatal("alias factory root must resolve")
	}
	if want := fac2.Scope().Lookup("Pub").Type(); got3 != want {
		t.Fatalf("alias resolution must preserve exact Alias identity: got %v, want %v", got3, want)
	}
}

func TestLogosOwnedFactoryRejectsPrivate(t *testing.T) {
	_, fac, _, _ := logosOwnedSyntheticDEF1(t)
	helperSig := NewSignatureType(nil, nil, nil, NewTuple(), NewTuple(NewParam(nopos, fac, "", Typ[Int])), false)
	fac.Scope().Insert(NewFunc(nopos, fac, "helper", helperSig))
	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": fac}}
	f := logosOwnedParseFile(t, "p.go", "package p\nvar _ T\nvar _ U\n")

	var msgs []string
	reg := map[syntax.Expr]LogosOwnedType{
		logosOwnedVarType(t, f, 0): LogosOwnedFactory("example.com/fac", "helper",
			LogosOwnedStep{Kind: LogosOwnedSigResult}),
		logosOwnedVarType(t, f, 1): LogosOwnedFactory("example.com/fac", "Nope",
			LogosOwnedStep{Kind: LogosOwnedSigResult}),
	}
	check := logosOwnedChecker(reg, imp, &msgs)
	for node := range reg {
		got, ok := check.logosOwnedTypeOf(node)
		if !ok {
			t.Fatal("registered node must report handled even on failure")
		}
		if got != Typ[Invalid] {
			t.Fatalf("private or missing factory must fail closed, got %v", got)
		}
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 diagnostics, got %d: %q", len(msgs), msgs)
	}
	for _, m := range msgs {
		if !strings.Contains(m, "logos:") {
			t.Fatalf("diagnostic must carry the logos prefix: %q", m)
		}
	}
}

func TestLogosOwnedMalformedDescriptor(t *testing.T) {
	_, fac, _, _ := logosOwnedSyntheticDEF1(t)
	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": fac}}
	f := logosOwnedParseFile(t, "p.go", "package p\nvar _ T\nvar _ U\nvar _ V\nvar _ W\n")
	n0 := logosOwnedVarType(t, f, 0)
	n1 := logosOwnedVarType(t, f, 1)
	n2 := logosOwnedVarType(t, f, 2)
	n3 := logosOwnedVarType(t, f, 3)
	wit := logosOwnedVarValue(t, logosOwnedParseFile(t, "w.go", "package p\nvar _ = 1\n"), 0)
	reg := map[syntax.Expr]LogosOwnedType{
		n0: {},
		n1: {FactoryPath: "example.com/fac", FactoryName: "Make", Witness: wit},
		n2: LogosOwnedFactory("example.com/fac", "Make"),
		n3: LogosOwnedFactory("example.com/fac", "Pub",
			LogosOwnedStep{Kind: LogosOwnedSigResult}),
	}
	var msgs []string
	check := logosOwnedChecker(reg, imp, &msgs)
	for node := range reg {
		if got, ok := check.logosOwnedTypeOf(node); !ok || got != Typ[Invalid] {
			t.Fatalf("malformed descriptor must fail closed: got %v, %v", got, ok)
		}
	}
	if len(msgs) != 4 {
		t.Fatalf("expected 4 diagnostics, got %d: %q", len(msgs), msgs)
	}
}

func TestLogosOwnedFactoryEndToEnd(t *testing.T) {
	facSrc := "package fac\ntype secret struct{ n int }\nfunc Make() map[string]secret { return nil }\nfunc Box[T any](v T) map[string]T { return nil }\n"
	mainSrc := "package main\nimport \"example.com/fac\"\nvar _ = fac.Make()\nvar _ = fac.Box(1)\n"
	sizes := SizesFor("gc", "amd64")
	facConf := &Config{GoVersion: "go1.23", Sizes: sizes, Context: NewContext()}
	facFile := logosOwnedParseFile(t, "fac.go", facSrc)
	facPkg, err := facConf.Check("example.com/fac", []*syntax.File{facFile}, &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{}, Selections: map[*syntax.SelectorExpr]*Selection{},
		Scopes: map[syntax.Node]*Scope{}, Instances: map[*syntax.Name]Instance{},
	})
	if err != nil {
		t.Fatalf("fac check: %v", err)
	}
	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": facPkg}}
	mainConf := &Config{GoVersion: "go1.23", Sizes: sizes, Importer: imp, Context: NewContext()}
	mainFile := logosOwnedParseFile(t, "main.go", mainSrc)
	mainInfo := &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{}, Selections: map[*syntax.SelectorExpr]*Selection{},
		Scopes: map[syntax.Node]*Scope{}, Instances: map[*syntax.Name]Instance{},
	}
	if _, err := mainConf.Check("example.com/main", []*syntax.File{mainFile}, mainInfo); err != nil {
		t.Fatalf("main check: %v", err)
	}
	w1 := logosOwnedVarValue(t, mainFile, 0)
	w2 := logosOwnedVarValue(t, mainFile, 1)
	mainCheck := NewChecker(mainConf, NewPackage("example.com/main", "main"), mainInfo)

	wantSecret := facPkg.Scope().Lookup("secret").Type()
	d1 := LogosOwnedWitness(w1, LogosOwnedStep{Kind: LogosOwnedMapElem})
	got1 := mainCheck.logosOwnedResolve(w1, d1)
	if got1 != wantSecret {
		t.Fatalf("witness must resolve the exact fresh Named: got %v, want %v", got1, wantSecret)
	}

	d2 := LogosOwnedWitness(w2, LogosOwnedStep{Kind: LogosOwnedMapElem})
	if got2 := mainCheck.logosOwnedResolve(w2, d2); got2 != Typ[Int] {
		t.Fatalf("instantiated witness must resolve fresh: got %v", got2)
	}

	typeNode := logosOwnedVarType(t, logosOwnedParseFile(t, "t.go", "package p\nvar _ T\n"), 0)
	reg := map[syntax.Expr]LogosOwnedType{
		typeNode: LogosOwnedFactory("example.com/fac", "Make",
			LogosOwnedStep{Kind: LogosOwnedSigResult},
			LogosOwnedStep{Kind: LogosOwnedMapElem}),
	}
	mainCheck.conf.LogosOwnedTypes = reg
	got3, ok := mainCheck.logosOwnedTypeOf(typeNode)
	if !ok {
		t.Fatal("factory type node must resolve")
	}
	if got3 != wantSecret {
		t.Fatalf("factory must resolve the exact fresh Named: got %v, want %v", got3, wantSecret)
	}
	if got3 != got1 {
		t.Fatal("factory and witness routes must agree on identity")
	}

	freshWit := logosOwnedVarValue(t, logosOwnedParseFile(t, "fresh.go", "package p\nvar _ = 1\n"), 0)
	if got := mainCheck.logosOwnedResolve(freshWit, LogosOwnedWitness(freshWit)); got != Typ[UntypedInt] {
		t.Fatalf("fresh witness must resolve on demand under a real checker: got %v", got)
	}
}

func TestLogosOwnedWitnessOnDemand(t *testing.T) {
	wit := logosOwnedVarValue(t, logosOwnedParseFile(t, "w.go", "package p\nvar _ = 1\n"), 0)
	var msgs []string
	check := logosOwnedChecker(nil, &logosOwnedTestImporter{}, &msgs)
	d := LogosOwnedWitness(wit)
	got := check.logosOwnedResolve(wit, d)
	if got != Typ[UntypedInt] {
		t.Fatalf("unchecked witness must be checked fresh on demand: got %v", got)
	}
	if len(msgs) != 0 {
		t.Fatalf("on-demand success must report nothing: %q", msgs)
	}

	mapT := NewMap(Typ[String], Typ[Int])
	pre := logosOwnedChecker(nil, &logosOwnedTestImporter{}, &msgs)
	pre.Info.Types = map[syntax.Expr]TypeAndValue{wit: {Type: mapT}}
	d2 := LogosOwnedWitness(wit, LogosOwnedStep{Kind: LogosOwnedMapElem})
	if got := pre.logosOwnedResolve(wit, d2); got != Typ[Int] {
		t.Fatalf("recorded witness must traverse fresh: got %v", got)
	}
}

func TestLogosOwnedWalkSteps(t *testing.T) {
	at := logosOwnedVarType(t, logosOwnedParseFile(t, "p.go", "package p\nvar _ T\n"), 0)
	var msgs []string
	check := logosOwnedChecker(nil, &logosOwnedTestImporter{}, &msgs)

	named := NewNamed(NewTypeName(nopos, NewPackage("example.com/d", "d"), "K", nil), Typ[Int], nil)
	secret := NewMap(Typ[String], named)
	st := NewStruct([]*Var{
		NewVar(nopos, nil, "a", Typ[Int]),
		NewVar(nopos, nil, "b", secret),
	}, nil)
	sig := NewSignatureType(nil, nil, nil,
		NewTuple(NewVar(nopos, nil, "p0", Typ[Int]), NewVar(nopos, nil, "p1", st)),
		NewTuple(NewVar(nopos, nil, "r0", secret)), false)
	aObj := NewTypeName(nopos, NewPackage("example.com/d", "d"), "A", nil)
	alias := NewAlias(aObj, secret)

	step := func(k LogosOwnedStepKind, i int) LogosOwnedStep { return LogosOwnedStep{Kind: k, Index: i} }
	type tc struct {
		root Type
		path []LogosOwnedStep
		want Type
		fail bool
	}
	cases := []tc{
		{named, []LogosOwnedStep{{Kind: LogosOwnedUnderlying}}, Typ[Int], false},
		{alias, []LogosOwnedStep{{Kind: LogosOwnedUnderlying}}, secret, false},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedMapKey}}, Typ[String], false},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedMapElem}}, named, false},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedMapElem}, {Kind: LogosOwnedUnderlying}}, Typ[Int], false},
		{NewSlice(named), []LogosOwnedStep{{Kind: LogosOwnedSliceElem}}, named, false},
		{NewArray(named, 3), []LogosOwnedStep{{Kind: LogosOwnedArrayElem}}, named, false},
		{NewPointer(named), []LogosOwnedStep{{Kind: LogosOwnedPointerElem}}, named, false},
		{st, []LogosOwnedStep{step(LogosOwnedStructField, 1)}, secret, false},
		{sig, []LogosOwnedStep{step(LogosOwnedSigResult, 0)}, secret, false},
		{sig, []LogosOwnedStep{step(LogosOwnedSigParam, 1)}, st, false},
		{sig, nil, Type(sig), false},
		{Typ[Int], []LogosOwnedStep{{Kind: LogosOwnedMapElem}}, nil, true},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedSliceElem}}, nil, true},
		{st, []LogosOwnedStep{step(LogosOwnedStructField, 2)}, nil, true},
		{sig, []LogosOwnedStep{step(LogosOwnedSigResult, 1)}, nil, true},
		{sig, []LogosOwnedStep{step(LogosOwnedSigParam, 9)}, nil, true},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedStepKind(0)}}, nil, true},
		{secret, []LogosOwnedStep{{Kind: LogosOwnedStepKind(99)}}, nil, true},
	}
	for i, c := range cases {
		msgs = nil
		got, ok := check.logosOwnedWalk(c.root, c.path, at)
		if c.fail {
			if ok || got != nil {
				t.Fatalf("case %d: must fail closed, got %v, %v", i, got, ok)
			}
			if len(msgs) == 0 {
				t.Fatalf("case %d: failure must carry a diagnostic", i)
			}
			continue
		}
		if !ok || got != c.want {
			t.Fatalf("case %d: got %v, %v; want %v", i, got, ok, c.want)
		}
	}
}

func TestLogosOwnedSelectorBypass(t *testing.T) {
	defPkg := NewPackage("example.com/def", "def")
	st := NewStruct([]*Var{
		NewVar(nopos, defPkg, "secret", Typ[Int]),
		NewVar(nopos, defPkg, "Pub", Typ[String]),
	}, nil)
	named := NewNamed(NewTypeName(nopos, defPkg, "S", nil), st, nil)

	selFile := logosOwnedParseFile(t, "s.go", "package p\nvar _ = v.secret\nvar _ = v.nope\n")
	selNode := logosOwnedVarValue(t, selFile, 0).(*syntax.SelectorExpr)
	missNode := logosOwnedVarValue(t, selFile, 1).(*syntax.SelectorExpr)
	reg := map[syntax.Expr]LogosOwnedType{selNode: {}, missNode: {}}
	var msgs []string
	check := logosOwnedChecker(reg, &logosOwnedTestImporter{}, &msgs)
	check.Info.Selections = map[*syntax.SelectorExpr]*Selection{}

	x := &operand{}
	x.mode_ = variable
	x.typ_ = named
	if !check.logosOwnedSelector(x, selNode) {
		t.Fatal("registered selector must be handled")
	}
	if x.typ() != Typ[Int] {
		t.Fatalf("selector must resolve the exact field type: got %v", x.typ())
	}
	sel, ok := check.Info.Selections[selNode]
	if !ok {
		t.Fatal("selector must record normal selection metadata")
	}
	if sel.Kind() != FieldVal || sel.Obj() != Object(st.Field(0)) {
		t.Fatalf("selection must name the exact field: %+v", sel)
	}
	if sel.Index() == nil || len(sel.Index()) != 1 || sel.Index()[0] != 0 {
		t.Fatalf("selection must carry the field index: %+v", sel)
	}
	if sel.Indirect() {
		t.Fatalf("direct selection must not mark indirection: %+v", sel)
	}

	dupFile := logosOwnedParseFile(t, "d.go", "package p\nvar _ = v.secret\n")
	dup := logosOwnedVarValue(t, dupFile, 0).(*syntax.SelectorExpr)
	y := &operand{}
	y.mode_ = variable
	y.typ_ = named
	if check.logosOwnedSelector(y, dup) {
		t.Fatal("same spelling on another node must keep normal checking")
	}

	z := &operand{}
	z.mode_ = variable
	z.typ_ = named
	if !check.logosOwnedSelector(z, missNode) {
		t.Fatal("registered selector must be handled even on failure")
	}
	if z.typ() != Typ[Invalid] {
		t.Fatalf("missing field must fail closed: got %v", z.typ())
	}

	p := &operand{}
	p.mode_ = value
	p.typ_ = NewPointer(named)
	if !check.logosOwnedSelector(p, selNode) {
		t.Fatal("pointer base must be handled")
	}
	if p.typ() != Typ[Int] || p.mode() != variable {
		t.Fatalf("pointer selection must deref to an addressable field: %v %v", p.typ(), p.mode())
	}
	if got := check.Info.Selections[selNode]; !got.Indirect() {
		t.Fatal("pointer selection must record indirection")
	}

	blank := &syntax.SelectorExpr{X: &syntax.Name{Value: "v"}, Sel: &syntax.Name{Value: "_"}}
	reg[blank] = LogosOwnedType{}
	q := &operand{}
	q.mode_ = variable
	q.typ_ = named
	if check.logosOwnedSelector(q, blank) {
		t.Fatal("blank selectors must keep normal checking")
	}

	badNode := logosOwnedVarValue(t, logosOwnedParseFile(t, "b.go", "package p\nvar _ = v.secret\n"), 0).(*syntax.SelectorExpr)
	var badMsgs []string
	badCheck := logosOwnedChecker(
		map[syntax.Expr]LogosOwnedType{badNode: LogosOwnedFactory("example.com/fac", "Make")},
		&logosOwnedTestImporter{}, &badMsgs)
	badCheck.Info.Selections = map[*syntax.SelectorExpr]*Selection{}
	b := &operand{}
	b.mode_ = variable
	b.typ_ = named
	if !badCheck.logosOwnedSelector(b, badNode) {
		t.Fatal("nonzero descriptor on a selector must be handled")
	}
	if b.typ() != Typ[Invalid] {
		t.Fatalf("nonzero descriptor on a selector must fail closed: got %v", b.typ())
	}
	if len(badMsgs) != 1 || !strings.Contains(badMsgs[0], "zero descriptor") {
		t.Fatalf("nonzero descriptor must carry a diagnostic: %q", badMsgs)
	}
}

func TestLogosOwnedLitFieldBypass(t *testing.T) {
	defPkg := NewPackage("example.com/def", "def")
	st := NewStruct([]*Var{
		NewVar(nopos, defPkg, "secret", Typ[Int]),
		NewVar(nopos, defPkg, "Pub", Typ[String]),
	}, nil)
	litFile := logosOwnedParseFile(t, "l.go", "package p\nvar _ = T{secret: 1}\n")
	lit := logosOwnedVarValue(t, litFile, 0).(*syntax.CompositeLit)
	reg := map[syntax.Expr]LogosOwnedType{lit: {}}
	check := logosOwnedChecker(reg, &logosOwnedTestImporter{}, nil)

	fld, idx, ok := check.logosOwnedLitField(lit, st, "secret", nil)
	if !ok || fld != st.Field(0) || len(idx) != 1 || idx[0] != 0 {
		t.Fatalf("registered literal must bypass the export check: %v %v %v", fld, idx, ok)
	}

	other := logosOwnedVarValue(t, logosOwnedParseFile(t, "m.go", "package p\nvar _ = T{secret: 1}\n"), 0).(*syntax.CompositeLit)
	if _, _, ok := check.logosOwnedLitField(other, st, "secret", nil); ok {
		t.Fatal("same spelling on another literal must not bypass")
	}

	accessible := NewVar(nopos, NewPackage("example.com/main", "main"), "Pub", Typ[String])
	if _, _, ok := check.logosOwnedLitField(lit, st, "Pub", accessible); ok {
		t.Fatal("accessible fields must keep normal checking")
	}

	if _, _, ok := check.logosOwnedLitField(lit, st, "nope", nil); ok {
		t.Fatal("missing fields must keep normal errors")
	}

	if _, _, ok := check.logosOwnedLitField(lit, nil, "secret", nil); ok {
		t.Fatal("nil struct must not bypass")
	}

	if _, _, ok := check.logosOwnedLitField(lit, st, "_", nil); ok {
		t.Fatal("blank literal fields must keep normal checking")
	}

	badLit := logosOwnedVarValue(t, logosOwnedParseFile(t, "bl.go", "package p\nvar _ = T{secret: 1}\n"), 0).(*syntax.CompositeLit)
	var badMsgs []string
	badCheck := logosOwnedChecker(
		map[syntax.Expr]LogosOwnedType{badLit: LogosOwnedFactory("example.com/fac", "Make")},
		&logosOwnedTestImporter{}, &badMsgs)
	fld, idx, ok = badCheck.logosOwnedLitField(badLit, st, "secret", nil)
	if !ok || fld != nil || idx != nil {
		t.Fatalf("nonzero descriptor on a literal must fail handled: %v %v %v", fld, idx, ok)
	}
	fld, idx, ok = badCheck.logosOwnedLitField(badLit, st, "Pub", accessible)
	if !ok || fld != nil || idx != nil {
		t.Fatalf("nonzero descriptor must fail even when the field is accessible: %v %v %v", fld, idx, ok)
	}
	if len(badMsgs) != 2 {
		t.Fatalf("both malformed uses must carry diagnostics: %q", badMsgs)
	}
	for _, m := range badMsgs {
		if !strings.Contains(m, "zero descriptor") {
			t.Fatalf("nonzero descriptor must carry a diagnostic: %q", m)
		}
	}
}

func logosOwnedSyntheticValuesDEF(t *testing.T) (facPkg *Package, privInt, privSecret *Named) {
	t.Helper()
	facPkg = NewPackage("example.com/fac", "fac")
	intObj := NewTypeName(nopos, facPkg, "privateInt", nil)
	privInt = NewNamed(intObj, Typ[Int], nil)
	facPkg.Scope().Insert(intObj)
	intSig := NewSignatureType(nil, nil, nil,
		NewTuple(NewParam(nopos, facPkg, "", Typ[Int])),
		NewTuple(NewParam(nopos, facPkg, "", privInt)), false)
	facPkg.Scope().Insert(NewFunc(nopos, facPkg, "MakePrivateInt", intSig))

	secObj := NewTypeName(nopos, facPkg, "privateSecret", nil)
	st := NewStruct([]*Var{
		NewVar(nopos, facPkg, "hiddenCode", Typ[Int]),
		NewVar(nopos, facPkg, "realm", Typ[String]),
	}, nil)
	privSecret = NewNamed(secObj, st, nil)
	facPkg.Scope().Insert(secObj)
	secSig := NewSignatureType(nil, nil, nil,
		NewTuple(NewParam(nopos, facPkg, "", Typ[Int]), NewParam(nopos, facPkg, "", Typ[String])),
		NewTuple(NewParam(nopos, facPkg, "", privSecret)), false)
	facPkg.Scope().Insert(NewFunc(nopos, facPkg, "MakePrivateSecret", secSig))
	return facPkg, privInt, privSecret
}

func logosOwnedFullCheck(t *testing.T, f *syntax.File, reg map[syntax.Expr]LogosOwnedType, imp *logosOwnedTestImporter) (*Info, []string, error) {
	t.Helper()
	sizes := SizesFor("gc", "amd64")
	var msgs []string
	conf := &Config{GoVersion: "go1.23", Sizes: sizes, Importer: imp, Context: NewContext(), LogosOwnedTypes: reg}
	conf.Error = func(err error) { msgs = append(msgs, err.Error()) }
	info := &Info{
		Types: map[syntax.Expr]TypeAndValue{}, Defs: map[*syntax.Name]Object{},
		Uses: map[*syntax.Name]Object{}, Selections: map[*syntax.SelectorExpr]*Selection{},
		Scopes: map[syntax.Node]*Scope{}, Instances: map[*syntax.Name]Instance{},
	}
	_, err := conf.Check("example.com/main", []*syntax.File{f}, info)
	return info, msgs, err
}

func TestLogosOwnedIdentConversionResolves(t *testing.T) {
	facPkg, privInt, _ := logosOwnedSyntheticValuesDEF(t)
	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": facPkg}}
	f := logosOwnedParseFile(t, "m.go", "package main\nvar _ = T(7)\n")
	call, ok := logosOwnedVarValue(t, f, 0).(*syntax.CallExpr)
	if !ok {
		t.Fatal("value must parse as a conversion call")
	}
	fun, ok := call.Fun.(*syntax.Name)
	if !ok {
		t.Fatal("conversion target must parse as a name")
	}
	reg := map[syntax.Expr]LogosOwnedType{
		fun: LogosOwnedFactory("example.com/fac", "MakePrivateInt",
			LogosOwnedStep{Kind: LogosOwnedSigResult, Index: 0}),
	}
	info, msgs, err := logosOwnedFullCheck(t, f, reg, imp)
	if err != nil {
		t.Fatalf("conversion through the owned placeholder must check: %v %q", err, msgs)
	}
	if len(msgs) != 0 {
		t.Fatalf("conversion must report nothing: %q", msgs)
	}
	tv, ok := info.Types[call]
	if !ok || tv.Type != Type(privInt) {
		t.Fatalf("conversion must record the exact fresh Named: got %v, %v", tv.Type, ok)
	}

	g := logosOwnedParseFile(t, "n.go", "package main\nvar _ = T(7)\n")
	if _, msgs, err := logosOwnedFullCheck(t, g, nil, imp); err == nil {
		t.Fatal("unregistered placeholder must not check")
	} else if len(msgs) == 0 || !strings.Contains(strings.Join(msgs, "\n"), "undefined: T") {
		t.Fatalf("unregistered placeholder must report undefined: %q", msgs)
	}
}

func TestLogosOwnedPositionalLiteralBypass(t *testing.T) {
	facPkg, _, privSecret := logosOwnedSyntheticValuesDEF(t)
	imp := &logosOwnedTestImporter{pkgs: map[string]*Package{"example.com/fac": facPkg}}
	f := logosOwnedParseFile(t, "m.go", "package main\nvar _ = T{42, \"vault\"}\n")
	lit, ok := logosOwnedVarValue(t, f, 0).(*syntax.CompositeLit)
	if !ok {
		t.Fatal("value must parse as a composite literal")
	}
	typeName, ok := lit.Type.(*syntax.Name)
	if !ok {
		t.Fatal("literal type must parse as a name")
	}
	reg := map[syntax.Expr]LogosOwnedType{
		typeName: LogosOwnedFactory("example.com/fac", "MakePrivateSecret",
			LogosOwnedStep{Kind: LogosOwnedSigResult, Index: 0}),
		lit: {},
	}
	info, msgs, err := logosOwnedFullCheck(t, f, reg, imp)
	if err != nil {
		t.Fatalf("registered positional literal must check: %v %q", err, msgs)
	}
	if len(msgs) != 0 {
		t.Fatalf("registered positional literal must report nothing: %q", msgs)
	}
	tv, ok := info.Types[lit]
	if !ok || tv.Type != Type(privSecret) {
		t.Fatalf("literal must record the exact fresh Named: got %v, %v", tv.Type, ok)
	}

	g := logosOwnedParseFile(t, "n.go", "package main\nvar _ = T{42, \"vault\"}\n")
	if _, msgs, err := logosOwnedFullCheck(t, g, nil, imp); err == nil {
		t.Fatal("unregistered literal must not check")
	} else if len(msgs) == 0 {
		t.Fatal("unregistered literal must report a diagnostic")
	}

	h := logosOwnedParseFile(t, "h.go", "package main\nvar _ = T{42, \"vault\"}\n")
	hlit := logosOwnedVarValue(t, h, 0).(*syntax.CompositeLit)
	htype, ok := hlit.Type.(*syntax.Name)
	if !ok {
		t.Fatal("literal type must parse as a name")
	}
	badReg := map[syntax.Expr]LogosOwnedType{
		htype: LogosOwnedFactory("example.com/fac", "MakePrivateSecret",
			LogosOwnedStep{Kind: LogosOwnedSigResult, Index: 0}),
		hlit: LogosOwnedFactory("example.com/fac", "MakePrivateSecret"),
	}
	if _, msgs, err := logosOwnedFullCheck(t, h, badReg, imp); err == nil {
		t.Fatal("nonzero literal descriptor must not check")
	} else if !strings.Contains(strings.Join(msgs, "\n"), "zero descriptor") {
		t.Fatalf("nonzero literal descriptor must carry the descriptor diagnostic: %q", msgs)
	}
}

func TestLogosOwnedPosLitFieldBypass(t *testing.T) {
	litFile := logosOwnedParseFile(t, "l.go", "package p\nvar _ = T{1}\n")
	lit := logosOwnedVarValue(t, litFile, 0).(*syntax.CompositeLit)
	reg := map[syntax.Expr]LogosOwnedType{lit: {}}
	check := logosOwnedChecker(reg, &logosOwnedTestImporter{}, nil)
	if !check.logosOwnedPosLitField(lit) {
		t.Fatal("registered literal must bypass the positional export check")
	}
	other := logosOwnedVarValue(t, logosOwnedParseFile(t, "m.go", "package p\nvar _ = T{1}\n"), 0).(*syntax.CompositeLit)
	if check.logosOwnedPosLitField(other) {
		t.Fatal("same spelling on another literal must not bypass")
	}
	bare := NewChecker(&Config{}, NewPackage("example.com/test", "test"), &Info{})
	if bare.logosOwnedPosLitField(lit) {
		t.Fatal("nil map must bypass nothing positionally")
	}
	if bare.logosOwnedPosLitField(nil) {
		t.Fatal("nil literal must not bypass")
	}
}

func TestLogosOwnedFirstCheckReceivesNothing(t *testing.T) {
	f := logosOwnedParseFile(t, "p.go", "package p\nvar _ T\n")
	node := logosOwnedVarType(t, f, 0)
	first := NewChecker(&Config{Logos: true}, NewPackage("example.com/test", "test"), &Info{})
	if _, ok := first.logosOwnedTypeOf(node); ok {
		t.Fatal("first check must resolve no owned nodes")
	}
	if first.logosOwnedRegistered(node) {
		t.Fatal("first check must register no owned nodes")
	}
}
