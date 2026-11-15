// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"cmd/internal/goobj"
)

func TestWriteConstFacts(t *testing.T) {
	const src = `package factstest

type Box struct {
	Items const []int
	Name  string
}

func Sum(xs const []int) int {
	total := 0
	for _, v := range xs {
		total += v
	}
	return total
}

func AppendIt(xs []int, v int) []int {
	return append(xs, v)
}

func Double(xs []int) []int {
	return AppendIt(AppendIt(xs, 1), 2)
}

func (b *Box) Rename(n string) {
	b.Name = n
}

func (b *Box) Count() int {
	return len(b.Items)
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
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
	pkg, _ := conf.Check("factstest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}

	got := facts.Write(pkg, info, []*syntax.File{file}, nil)

	consts := make(map[string]bool)
	for _, c := range got.Consts {
		consts[c.Object+" "+c.Path] = true
	}
	for _, want := range []string{"Sum P0", "Box FItems"} {
		if !consts[want] {
			t.Errorf("missing const line %q (have %v)", want, got.Consts)
		}
	}

	fns := make(map[string]facts.FnFact)
	for _, fn := range got.Fns {
		fns[fn.Object] = fn
	}
	checkFn := func(object, mask string, recv bool) {
		t.Helper()
		fn, ok := fns[object]
		if mask == "" && !recv {
			if ok {
				t.Errorf("%s unexpectedly has facts %+v", object, fn)
			}
			return
		}
		if !ok {
			t.Errorf("missing fn facts for %s", object)
			return
		}
		if fn.ParamMutated != mask {
			t.Errorf("%s ParamMutated = %q, want %q", object, fn.ParamMutated, mask)
		}
		if fn.ReceiverMutated != recv {
			t.Errorf("%s ReceiverMutated = %v, want %v", object, fn.ReceiverMutated, recv)
		}
	}
	checkFn("AppendIt", "1", false)
	checkFn("Double", "1", false)
	checkFn("(*Box).Rename", "", true)
	checkFn("Sum", "", false)
	checkFn("(*Box).Count", "", false)
}

func TestWriteCheckedAssertEffects(t *testing.T) {
	const src = `package asserttest

type Qual interface {
	const Name() string
}

func Identity[T any](v T) T { return v }

func As[T any](x any) T { return x.(T) }

type Box[T any] struct{ V T }

func (b Box[T]) As(x any) T { return x.(T) }

type Store[T any] struct{ V T }

func (s Store[T]) Get() T { return s.V }

func (s Store[T]) Project[U any](x any) U { return x.(U) }

type Plain struct{}
func (Plain) Identity[U any](x U) U { return x }
func (Plain) As[U any](x any) U { return x.(U) }
`
	dir := t.TempDir()
	path := filepath.Join(dir, "assert.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
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
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	pkg, _ := conf.Check("asserttest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check (definitions must stay valid): %v", err)
	}

	seed := &facts.Facts{Fns: []facts.FnFact{{Object: "As", ParamMutated: "1", Readonly: true}}}
	got := facts.WriteChecked(pkg, info, []*syntax.File{file}, seed)
	if got != seed {
		t.Fatal("WriteChecked did not return the passed facts")
	}
	fns := make(map[string]facts.FnFact)
	for _, fn := range got.Fns {
		fns[fn.Object] = fn
	}
	check := func(object, tp, rtp string) {
		t.Helper()
		fn, ok := fns[object]
		if !ok || !fn.HasAssert {
			t.Errorf("%s: missing Known assert record (have %+v)", object, fn)
			return
		}
		if fn.AssertTP != tp || fn.AssertRTP != rtp {
			t.Errorf("%s: AssertTP=%q AssertRTP=%q, want %q/%q", object, fn.AssertTP, fn.AssertRTP, tp, rtp)
		}
	}
	check("Identity", "0", "0")
	check("As", "1", "0")
	check("Box.As", "0", "1")
	check("Store.Get", "0", "0")
	check("Store.Project", "1", "0")
	check("Plain.Identity", "0", "0")
	check("Plain.As", "1", "0")
	if fn := fns["As"]; !fn.Readonly || fn.ParamMutated != "1" {
		t.Errorf("As lost seeded flags: %+v", fn)
	}
	if out := facts.WriteChecked(pkg, info, []*syntax.File{file}, nil); out == nil {
		t.Fatal("WriteChecked returned nil for a generic package")
	}
	nogen := `package nogen

func F() int { return 1 }
`
	ndir := t.TempDir()
	npath := filepath.Join(ndir, "nogen.lgo")
	if err := os.WriteFile(npath, []byte(nogen), 0o644); err != nil {
		t.Fatal(err)
	}
	nfile, err := syntax.ParseFile(npath, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	ninfo := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	var nerrs []error
	nconf := types2.Config{
		Importer: imp,
		Error:    func(err error) { nerrs = append(nerrs, err) },
		Logos:    true,
	}
	npkg, _ := nconf.Check("nogen", []*syntax.File{nfile}, ninfo)
	if err := errors.Join(nerrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	if out := facts.WriteChecked(npkg, ninfo, []*syntax.File{nfile}, nil); out != nil {
		t.Fatalf("WriteChecked on a package with no generic declaration = %+v, want nil", out)
	}
}

func TestApplyFuncFactsReachablePrivateOrigin(t *testing.T) {
	pkg := types2.NewPackage("example/privatefacts", "privatefacts")
	pos := syntax.Pos{}
	any := types2.NewInterfaceType(nil, nil)
	tp := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "T", nil), any)
	private := types2.NewNamed(types2.NewTypeName(pos, pkg, "private", nil), types2.NewStruct(nil, nil), nil)
	private.SetTypeParams([]*types2.TypeParam{tp})
	rp := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "U", nil), any)
	recvType, err := types2.Instantiate(nil, private, []types2.Type{rp}, false)
	if err != nil {
		t.Fatal(err)
	}
	recv := types2.NewVar(pos, pkg, "self", recvType)
	method := types2.NewFunc(pos, pkg, "As", types2.NewSignatureType(recv, []*types2.TypeParam{rp}, nil, nil, nil, false))
	private.AddMethod(method)
	instance, err := types2.Instantiate(nil, private, []types2.Type{types2.Typ[types2.Int]}, false)
	if err != nil {
		t.Fatal(err)
	}
	factory := types2.NewFunc(pos, pkg, "Factory", types2.NewSignatureType(nil, nil, nil, nil,
		types2.NewTuple(types2.NewVar(pos, pkg, "", instance)), false))
	pkg.Scope().Insert(factory)
	if pkg.Scope().Lookup("private") != nil {
		t.Fatal("private type must be reachable only through Factory's result")
	}
	before := factory.Type()
	records := []facts.FnFact{{Object: "private.As", HasAssert: true, AssertTP: "0", AssertRTP: "1", Readonly: true}}
	if err := ApplyFuncFacts(pkg, records); err != nil {
		t.Fatal(err)
	}
	if pkg.Scope().Lookup("Factory") != factory || factory.Type() != before {
		t.Fatal("metadata stamping replaced canonical declaration or signature")
	}
	fp, fr, known := method.LogosAssertEffects()
	if !known || types2.LogosAssertMaskHex(fp) != "0" || types2.LogosAssertMaskHex(fr) != "1" || !method.LogosReadonly() {
		t.Fatal("reachable private method origin lost verified metadata")
	}
	cloned, err := CloneWithOverlaysAndReadonly(pkg, nil, nil, records, nil, nil)
	if err != nil || cloned != pkg {
		t.Fatalf("declaration-only clone API replaced package: %p != %p (%v)", cloned, pkg, err)
	}
}

func TestApplyFuncFactsRejectsMaskOutsideArity(t *testing.T) {
	for _, tc := range []struct{ name, tp, rtp string }{
		{"extra function bit", "2", "0"},
		{"receiver bit on function", "0", "1"},
		{"partial pair", "0", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pkg := types2.NewPackage("example/arityfacts", "arityfacts")
			pos := syntax.Pos{}
			any := types2.NewInterfaceType(nil, nil)
			tp := types2.NewTypeParam(types2.NewTypeName(pos, pkg, "T", nil), any)
			fn := types2.NewFunc(pos, pkg, "Identity", types2.NewSignatureType(nil, nil, []*types2.TypeParam{tp}, nil, nil, false))
			pkg.Scope().Insert(fn)
			err := ApplyFuncFacts(pkg, []facts.FnFact{{Object: "Identity", HasAssert: true, AssertTP: tc.tp, AssertRTP: tc.rtp}})
			if err == nil {
				t.Fatal("malformed summary accepted at import boundary")
			}
			if _, _, known := fn.LogosAssertEffects(); known {
				t.Fatal("failed validation published a trusted summary")
			}
		})
	}
}

func TestCopyFuncReadonlyPropagatesBit(t *testing.T) {
	pkg := types2.NewPackage("example/readonlycopy", "readonlycopy")
	pos := syntax.Pos{}
	sig := types2.NewSignatureType(nil, nil, nil, nil, nil, false)
	src := types2.NewFunc(pos, pkg, "Src", sig)
	dst := types2.NewFunc(pos, pkg, "Dst", sig)
	src.SetLogosReadonly(true)
	copyFuncReadonly(dst, src)
	if !dst.LogosReadonly() {
		t.Fatal("clone lost the readonly bit")
	}
	plain := types2.NewFunc(pos, pkg, "Plain", sig)
	clean := types2.NewFunc(pos, pkg, "Clean", sig)
	copyFuncReadonly(clean, plain)
	if clean.LogosReadonly() {
		t.Fatal("clone invented a readonly bit")
	}
	copyFuncReadonly(nil, src)
	copyFuncReadonly(dst, nil)
}

func TestEmitGoParamMutatedFacts(t *testing.T) {
	const src = `package gopkg

func ReadAll(b []byte) int {
	n := 0
	for _, v := range b {
		n += int(v)
	}
	return n
}

func WriteAt(b []byte, v byte) {
	b[0] = v
}

func fill(b []byte) {
	b[0] = 0
}

func CopyInto(dst []byte) {
	fill(dst)
}

func Id[T any](x T) T {
	return x
}

func Ext(b []byte)

func GetItems() []int {
	return []int{1, 2, 3}
}

type Counter struct {
	N     int
	saved []byte
}

func (c Counter) Total(xs []int) int {
	n := c.N
	for _, v := range xs {
		n += v
	}
	return n
}

func (c *Counter) Fill(xs []byte, v byte) {
	for i := range xs {
		xs[i] = v
	}
}

func (c *Counter) Stash(xs []byte) {
	c.saved = xs
}

func (c *Counter) SetN(v int) {
	c.N = v
}

func Wrap(xs []byte) [][]byte {
	tmp := make([][]byte, 1)
	tmp[0] = xs
	return tmp
}

func (c *Counter) Bump() {
	c.N++
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.go")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	imp := newTestImporter(t)
	var checkErrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { checkErrs = append(checkErrs, err) },
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	pkg, _ := conf.Check("gopkg", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	got := facts.EmitGoMutations(pkg, info, []*syntax.File{file}, nil)
	if got == nil {
		t.Fatal("EmitGoMutations returned nil")
	}
	if got.GoPackage {
		t.Error("emitter sets the lang go record; the importer hook owns it")
	}
	for _, kind := range []int{
		len(got.Opts), len(got.Consts), len(got.Readonlys), len(got.Props),
		len(got.Conforms), len(got.Comptimes), len(got.ConstFuncs),
		len(got.Defaults), len(got.Locks), len(got.AllocFrees),
	} {
		if kind != 0 {
			t.Fatalf("Go member carries declaration-header records: %+v", got)
		}
	}
	masks := make(map[string]string)
	for _, fn := range got.Fns {
		masks[fn.Object] = fn.ParamMutated
	}
	want := map[string]string{
		"ReadAll":          "0",
		"WriteAt":          "1",
		"CopyInto":         "1",
		"Counter.Total":    "0",
		"(*Counter).Fill":  "1",
		"(*Counter).Stash": "1",
		"(*Counter).SetN":  "0",
		"Wrap":             "1",
	}
	if len(masks) != len(want) {
		t.Fatalf("fn entries = %v, want %v", masks, want)
	}
	for object, mask := range want {
		if masks[object] != mask {
			t.Errorf("%s ParamMutated = %q, want %q (have %v)", object, masks[object], mask, masks)
		}
	}
	fp := goobj.FingerprintType{1, 2, 3, 4, 5, 6, 7, 8}
	back, err := facts.Decode(facts.Encode(fp, got))
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}
	if len(back.Fns) != len(want) {
		t.Fatalf("round trip fn entries = %v, want %v", back.Fns, want)
	}
	for _, fn := range back.Fns {
		if want[fn.Object] != fn.ParamMutated {
			t.Errorf("round trip %s ParamMutated = %q, want %q", fn.Object, fn.ParamMutated, want[fn.Object])
		}
	}
}
