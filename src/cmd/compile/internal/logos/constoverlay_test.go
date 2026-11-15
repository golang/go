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
)

func TestCloneWithOverlaysComposesQualifiers(t *testing.T) {
	const src = `package overlaytest

type privateSlice []int

func (s privateSlice) Size() int {
	return len(s)
}

func View() const privateSlice {
	values := privateSlice{1, 2, 3, 4, 5}
	return values[:3]
}

func Sum(xs const []int) int {
	total := 0
	for _, v := range xs {
		total += v
	}
	return total
}

type Box struct {
	Items const []int
	Name  string
}

func (b *Box) Count() int {
	return len(b.Items)
}

func (b *Box) Total(v const []int) int {
	n := len(b.Items)
	for _, x := range v {
		n += x
	}
	return n
}

type BoxAlias = Box

func UseBoxAlias(a BoxAlias) int {
	return len(a.Items)
}

func TakeBox(b Box) int {
	return len(b.Items)
}

func MakeBox() Box {
	return Box{Name: "n"}
}

type List struct {
	Next *List
	Vals const []int
}

func (l *List) Len() int {
	return len(l.Vals)
}

func Find(haystack []string, needle string) []int? {
	return nil
}

type Store interface {
	Get(k string) const []byte
	Len() int
}

func First[T any](xs []T) T {
	return xs[0]
}

type Alias = []int

func UseAlias(a Alias) int {
	return len(a)
}

func Plain(x int) int {
	return x
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(opts) == 0 || len(consts) == 0 {
		t.Fatalf("want both opt and const facts, got %d opts and %d consts", len(opts), len(consts))
	}
	for _, want := range [][2]string{
		{"Box", "FItems"},
		{"TakeBox", "P0.FItems"},
		{"MakeBox", "R0.FItems"},
		{"List", "FVals"},
	} {
		if !hasPath(consts, want[0], want[1]) {
			t.Fatalf("missing const fact %s %s in %v", want[0], want[1], consts)
		}
	}

	cloned := CloneWithOverlays(lowered, opts, consts)
	if cloned == lowered {
		t.Fatal("CloneWithOverlays returned the input package")
	}
	if cloned.Path() != lowered.Path() || cloned.Name() != lowered.Name() {
		t.Fatalf("clone identity: got %s %s, want %s %s", cloned.Path(), cloned.Name(), lowered.Path(), lowered.Name())
	}

	viewOld := funcResult(t, lowered, "View", 0)
	if constAsView(viewOld) != nil {
		t.Fatal("lowered View result unexpectedly carries a const view")
	}
	viewNew := funcResult(t, cloned, "View", 0)
	c := constAsView(viewNew)
	if c == nil {
		t.Fatalf("cloned View result is %T, want *ConstView", viewNew)
	}
	named, ok := types2.Unalias(c.Elem).(*types2.Named)
	if !ok || named.Obj().Name() != "privateSlice" {
		t.Fatalf("View view element is %v, want named privateSlice", c.Elem)
	}
	if c.ConstMap {
		t.Fatal("restored view invents ConstMap the facts do not encode")
	}
	foundSize := false
	for i := range named.NumMethods() {
		if named.Method(i).Name() == "Size" {
			foundSize = true
		}
	}
	if !foundSize {
		t.Fatal("cloned View element lost method Size")
	}

	sumNew := funcParam(t, cloned, "Sum", 0)
	if constAsView(sumNew) == nil {
		t.Fatalf("cloned Sum param is %T, want *ConstView", sumNew)
	}
	itemsNew := structField(t, cloned, "Box", "Items")
	if constAsView(itemsNew) == nil {
		t.Fatalf("cloned Box.Items is %T, want *ConstView", itemsNew)
	}
	if got := structField(t, cloned, "Box", "Name"); constAsView(got) != nil {
		t.Fatalf("cloned Box.Name is a const view, want plain %v", got)
	}
	if m := lookupMethod(t, cloned, "Box", "Count"); m == nil {
		t.Fatal("cloned Box lost method Count")
	}
	totalNew := methodParam(t, cloned, "Box", "Total", 0)
	if constAsView(totalNew) == nil {
		t.Fatalf("cloned (*Box).Total param is %T, want *ConstView", totalNew)
	}

	oldBox := lowered.Scope().Lookup("Box").(*types2.TypeName).Type().(*types2.Named)
	newBoxTN, ok := cloned.Scope().Lookup("Box").(*types2.TypeName)
	if !ok {
		t.Fatal("cloned Box is not a type name")
	}
	newBox, ok := newBoxTN.Type().(*types2.Named)
	if !ok {
		t.Fatal("cloned Box is not named")
	}
	if newBox == oldBox {
		t.Fatal("Box shared, want rebuilt shell")
	}
	countSig, ok := lookupMethod(t, cloned, "Box", "Count").Type().(*types2.Signature)
	if !ok {
		t.Fatal("cloned (*Box).Count has no signature")
	}
	countRecv, ok := types2.Unalias(countSig.Recv().Type()).(*types2.Pointer)
	if !ok {
		t.Fatalf("cloned (*Box).Count recv is %T, want pointer", countSig.Recv().Type())
	}
	if recvBase, ok := types2.Unalias(countRecv.Elem()).(*types2.Named); !ok || recvBase != newBox {
		t.Fatalf("cloned (*Box).Count recv base is %v, want the cloned Box", countRecv.Elem())
	}

	if cloned.Scope().Lookup("TakeBox") == lowered.Scope().Lookup("TakeBox") {
		t.Fatal("TakeBox shared, want rebuilt around the cloned Box")
	}
	takeParam, ok := types2.Unalias(funcParam(t, cloned, "TakeBox", 0)).(*types2.Named)
	if !ok || takeParam != newBox {
		t.Fatalf("cloned TakeBox param is %v, want the cloned Box", funcParam(t, cloned, "TakeBox", 0))
	}
	if cloned.Scope().Lookup("MakeBox") == lowered.Scope().Lookup("MakeBox") {
		t.Fatal("MakeBox shared, want rebuilt around the cloned Box")
	}
	makeResult, ok := types2.Unalias(funcResult(t, cloned, "MakeBox", 0)).(*types2.Named)
	if !ok || makeResult != newBox {
		t.Fatalf("cloned MakeBox result is %v, want the cloned Box", funcResult(t, cloned, "MakeBox", 0))
	}

	oldAlias := lowered.Scope().Lookup("BoxAlias").(*types2.TypeName).Type().(*types2.Alias)
	newAliasTN, ok := cloned.Scope().Lookup("BoxAlias").(*types2.TypeName)
	if !ok {
		t.Fatal("cloned BoxAlias is not a type name")
	}
	newAlias, ok := newAliasTN.Type().(*types2.Alias)
	if !ok {
		t.Fatal("cloned BoxAlias is not an alias")
	}
	if newAlias == oldAlias {
		t.Fatal("BoxAlias shared, want rebuilt around the cloned Box")
	}
	aliasRHS, ok := types2.Unalias(newAlias.Rhs()).(*types2.Named)
	if !ok || aliasRHS.Origin() != newBox {
		t.Fatalf("cloned BoxAlias RHS is %v, want the cloned Box", newAlias.Rhs())
	}
	if cloned.Scope().Lookup("UseBoxAlias") == lowered.Scope().Lookup("UseBoxAlias") {
		t.Fatal("UseBoxAlias shared, want rebuilt around the cloned alias")
	}
	if got := funcParam(t, cloned, "UseBoxAlias", 0); got != types2.Type(newAlias) {
		t.Fatalf("cloned UseBoxAlias param is %v, want the cloned BoxAlias", got)
	}

	oldList := lowered.Scope().Lookup("List").(*types2.TypeName).Type().(*types2.Named)
	newListTN, ok := cloned.Scope().Lookup("List").(*types2.TypeName)
	if !ok {
		t.Fatal("cloned List is not a type name")
	}
	newList, ok := newListTN.Type().(*types2.Named)
	if !ok {
		t.Fatal("cloned List is not named")
	}
	if newList == oldList {
		t.Fatal("List shared, want rebuilt shell")
	}
	nextField := structField(t, cloned, "List", "Next")
	nextPtr, ok := types2.Unalias(nextField).(*types2.Pointer)
	if !ok {
		t.Fatalf("cloned List.Next is %T, want pointer", nextField)
	}
	if nextBase, ok := types2.Unalias(nextPtr.Elem()).(*types2.Named); !ok || nextBase != newList {
		t.Fatalf("cloned List.Next points at %v, want the same cloned List", nextPtr.Elem())
	}
	if lookupMethod(t, cloned, "List", "Len") == nil {
		t.Fatal("cloned List lost method Len")
	}

	findNew := funcResult(t, cloned, "Find", 0)
	if optAsOption(findNew) == nil {
		t.Fatalf("cloned Find result is %T, want *Option", findNew)
	}

	getNew := ifaceMethodResult(t, cloned, "Store", "Get", 0)
	if constAsView(getNew) == nil {
		t.Fatalf("cloned Store.Get result is %T, want *ConstView", getNew)
	}
	lenOld := ifaceMethodResult(t, lowered, "Store", "Len", 0)
	lenNew := ifaceMethodResult(t, cloned, "Store", "Len", 0)
	if !types2.Identical(lenOld, lenNew) {
		t.Fatalf("Store.Len changed: %v vs %v", lenOld, lenNew)
	}

	for _, name := range []string{"Plain", "First", "UseAlias", "Alias"} {
		if cloned.Scope().Lookup(name) != lowered.Scope().Lookup(name) {
			t.Fatalf("%s was copied, want shared", name)
		}
	}
	if cloned.Scope().Lookup("View") == lowered.Scope().Lookup("View") {
		t.Fatal("View was shared, want rebuilt with const view")
	}
}

func TestCloneWithOverlayKeepsOptsOnly(t *testing.T) {
	const src = `package overlaytest

func Find(haystack []string) []int? {
	return nil
}

func Sum(xs const []int) int {
	total := 0
	for _, v := range xs {
		total += v
	}
	return total
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(opts) == 0 || len(consts) == 0 {
		t.Fatalf("want both opt and const facts, got %d opts and %d consts", len(opts), len(consts))
	}
	cloned := CloneWithOverlays(lowered, opts, nil)
	if optAsOption(funcResult(t, cloned, "Find", 0)) == nil {
		t.Fatal("CloneWithOverlays did not restore the option result")
	}
	if constAsView(funcParam(t, cloned, "Sum", 0)) != nil {
		t.Fatal("CloneWithOverlays unexpectedly restored a const view")
	}
}

func TestCloneWithOverlaysGenericMethod(t *testing.T) {
	const src = `package overlaytest

type GBox[T any] struct {
	V T
}

func (b GBox[T]) Get() const T {
	return b.V
}

func (b GBox[T]) Len() int {
	return 0
}
`
	lowered, _, consts := checkLoweredForOverlay(t, src)
	if len(consts) == 0 {
		t.Fatal("want const facts")
	}
	cloned := CloneWithOverlays(lowered, nil, consts)
	oldTN, ok := lowered.Scope().Lookup("GBox").(*types2.TypeName)
	if !ok {
		t.Fatal("lowered GBox is not a type name")
	}
	newTN, ok := cloned.Scope().Lookup("GBox").(*types2.TypeName)
	if !ok {
		t.Fatal("cloned GBox is not a type name")
	}
	if newTN == oldTN {
		t.Fatal("generic GBox shared, want rebuilt")
	}
	newNamed, ok := newTN.Type().(*types2.Named)
	if !ok {
		t.Fatal("cloned GBox is not named")
	}
	if tp := newNamed.TypeParams(); tp == nil || tp.Len() != 1 {
		t.Fatalf("cloned GBox has %v type params, want 1", tp)
	}
	newGet := lookupMethod(t, cloned, "GBox", "Get")
	newSig, ok := newGet.Type().(*types2.Signature)
	if !ok {
		t.Fatal("cloned Get has no signature")
	}
	newView := constAsView(newSig.Results().At(0).Type())
	if newView == nil {
		t.Fatal("cloned GBox.Get result is not a const view")
	}
	newT, ok := newView.Elem.(*types2.TypeParam)
	if !ok || newT.Obj().Name() != "T" {
		t.Fatalf("view element is %v, want type param T", newView.Elem)
	}
	if rp := newSig.RecvTypeParams(); rp == nil || rp.Len() != 1 || newT != rp.At(0) {
		t.Fatal("view element is not the cloned method's own fresh receiver type parameter")
	}
	oldGet := lookupMethod(t, lowered, "GBox", "Get")
	oldSig, ok := oldGet.Type().(*types2.Signature)
	if !ok {
		t.Fatal("lowered Get has no signature")
	}
	oldT := oldSig.Results().At(0).Type()
	if constAsView(oldT) != nil {
		t.Fatal("lowered Get result unexpectedly carries a const view")
	}
	if newT == oldT {
		t.Fatal("view element reuses the old type parameter")
	}
	if orp := oldSig.RecvTypeParams(); orp != nil && orp.Len() == 1 && newT == orp.At(0) {
		t.Fatal("view element reuses the old method receiver type parameter")
	}
	recv := types2.Unalias(newSig.Recv().Type())
	inst, ok := recv.(*types2.Named)
	if !ok || inst.Origin() != newNamed {
		t.Fatalf("recv is %T (%v), want value instance of the cloned GBox", newSig.Recv().Type(), newSig.Recv().Type())
	}
	if inst.TypeArgs().Len() != 1 || inst.TypeArgs().At(0) != types2.Type(newT) {
		t.Fatalf("recv instance args are %v, want the clone's fresh T", inst.TypeArgs())
	}
	if lookupMethod(t, cloned, "GBox", "Len") == nil {
		t.Fatal("cloned GBox lost method Len")
	}
}

func TestCloneWithOverlaysNestedQualifiers(t *testing.T) {
	const src = `package overlaytest

func Both(xs const []int?) int {
	return 0
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(opts) == 0 || len(consts) == 0 {
		t.Fatalf("want both opt and const facts, got %d opts and %d consts", len(opts), len(consts))
	}
	if !hasPath(consts, "Both", "P0") {
		t.Fatalf("missing const fact Both P0 in %v", consts)
	}
	if !hasPath(opts, "Both", "P0.E") {
		t.Fatalf("missing opt fact Both P0.E in %v", opts)
	}
	cloned := CloneWithOverlays(lowered, opts, consts)
	got := funcParam(t, cloned, "Both", 0)
	view := constAsView(got)
	if view == nil {
		t.Fatalf("cloned Both param is %T, want *ConstView", got)
	}
	opt, ok := view.Elem.(*types2.Option)
	if !ok {
		t.Fatalf("view element is %T, want *Option", view.Elem)
	}
	if _, ok := opt.Elem.(*types2.Slice); !ok {
		t.Fatalf("option element is %T, want slice", opt.Elem)
	}
}

func TestCloneWithOverlaysPointerChannel(t *testing.T) {
	const src = `package overlaytest

func PtrOpt(p *([]int?)) int {
	return 0
}

func ChanOpt(c chan ([]int?)) int {
	return 0
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(opts) == 0 {
		t.Fatalf("want opt facts, got %d opts and %d consts", len(opts), len(consts))
	}
	for _, want := range [][2]string{
		{"PtrOpt", "P0.D"},
		{"ChanOpt", "P0.D"},
	} {
		if !hasPath(opts, want[0], want[1]) {
			t.Fatalf("missing opt fact %s %s in %v", want[0], want[1], opts)
		}
	}
	cloned := CloneWithOverlays(lowered, opts, consts)

	if cloned.Scope().Lookup("PtrOpt") == lowered.Scope().Lookup("PtrOpt") {
		t.Fatal("PtrOpt shared, want rebuilt with option behind pointer")
	}
	pOld := types2.Unalias(funcParam(t, lowered, "PtrOpt", 0))
	if _, ok := pOld.(*types2.Pointer); !ok {
		t.Fatalf("lowered PtrOpt param is %T, want pointer", funcParam(t, lowered, "PtrOpt", 0))
	}
	pNew := types2.Unalias(funcParam(t, cloned, "PtrOpt", 0))
	pPtr, ok := pNew.(*types2.Pointer)
	if !ok {
		t.Fatalf("cloned PtrOpt param is %T, want pointer", funcParam(t, cloned, "PtrOpt", 0))
	}
	pOpt := optAsOption(pPtr.Elem())
	if pOpt == nil {
		t.Fatalf("pointer element is %T, want *Option", pPtr.Elem())
	}
	if _, ok := pOpt.Elem.(*types2.Slice); !ok {
		t.Fatalf("option element behind pointer is %T, want slice", pOpt.Elem)
	}

	if cloned.Scope().Lookup("ChanOpt") == lowered.Scope().Lookup("ChanOpt") {
		t.Fatal("ChanOpt shared, want rebuilt with option behind channel")
	}
	cOld := types2.Unalias(funcParam(t, lowered, "ChanOpt", 0))
	if _, ok := cOld.(*types2.Chan); !ok {
		t.Fatalf("lowered ChanOpt param is %T, want channel", funcParam(t, lowered, "ChanOpt", 0))
	}
	cNew := types2.Unalias(funcParam(t, cloned, "ChanOpt", 0))
	cChan, ok := cNew.(*types2.Chan)
	if !ok {
		t.Fatalf("cloned ChanOpt param is %T, want channel", funcParam(t, cloned, "ChanOpt", 0))
	}
	cOpt := optAsOption(cChan.Elem())
	if cOpt == nil {
		t.Fatalf("channel element is %T, want *Option", cChan.Elem())
	}
	if _, ok := cOpt.Elem.(*types2.Slice); !ok {
		t.Fatalf("option element behind channel is %T, want slice", cOpt.Elem)
	}
}

func TestCloneWithOverlaysAnonStructAndFunc(t *testing.T) {
	const src = `package overlaytest

func Anon(s struct {
	F const []int
	G string
}) int {
	return len(s.F)
}

func WithCb(cb func(x const int) int) []int? {
	return nil
}
`
	lowered, opts, consts := checkLoweredForOverlay(t, src)
	if len(consts) == 0 || len(opts) == 0 {
		t.Fatalf("want const and opt facts, got %d consts and %d opts", len(consts), len(opts))
	}
	if !hasPath(consts, "WithCb", "P0.P0") {
		t.Fatalf("missing const fact WithCb P0.P0 in %v", consts)
	}
	if !hasPath(opts, "WithCb", "R0") {
		t.Fatalf("missing opt fact WithCb R0 in %v", opts)
	}
	cloned := CloneWithOverlays(lowered, opts, consts)
	s := funcParam(t, cloned, "Anon", 0)
	st, ok := types2.Unalias(s).(*types2.Struct)
	if !ok {
		t.Fatalf("cloned Anon param is %T, want struct", s)
	}
	for i := range st.NumFields() {
		f := st.Field(i)
		if f.Name() == "F" && constAsView(f.Type()) == nil {
			t.Fatalf("cloned Anon.F is %T, want *ConstView", f.Type())
		}
		if f.Name() == "G" && constAsView(f.Type()) != nil {
			t.Fatalf("cloned Anon.G is a const view, want plain %v", f.Type())
		}
	}
	cb := funcParam(t, cloned, "WithCb", 0)
	sig, ok := types2.Unalias(cb).(*types2.Signature)
	if !ok {
		t.Fatalf("cloned WithCb param is %T, want signature", cb)
	}
	if constAsView(sig.Params().At(0).Type()) == nil {
		t.Fatal("cloned callback param is not a const view")
	}
	if opt := optAsOption(funcResult(t, cloned, "WithCb", 0)); opt == nil {
		t.Fatalf("cloned WithCb result is %T, want *Option", funcResult(t, cloned, "WithCb", 0))
	}
}

func hasPath(facts []facts.PathFact, object, path string) bool {
	for _, f := range facts {
		if f.Object == object && f.Path == path {
			return true
		}
	}
	return false
}

func checkLoweredForOverlay(t *testing.T, src string) (*types2.Package, []facts.PathFact, []facts.PathFact) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "overlay.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	var perrs []error
	file, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	imp := newTestImporter(t)
	newInfo := func() *types2.Info {
		return &types2.Info{
			Types:      make(map[syntax.Expr]types2.TypeAndValue),
			Defs:       make(map[*syntax.Name]types2.Object),
			Uses:       make(map[*syntax.Name]types2.Object),
			Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
			Scopes:     make(map[syntax.Node]*types2.Scope),
		}
	}
	info := newInfo()
	var cerrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { cerrs = append(cerrs, err) },
		Logos:    true,
	}
	files := []*syntax.File{file}
	pkg, _ := conf.Check("overlaytest", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	f := facts.Write(pkg, info, files, map[*syntax.File][]byte{file: []byte(src)})
	if len(f.Opts)+len(f.Consts) == 0 {
		t.Fatal("no facts emitted")
	}
	changed, edits := Lower(pkg, info, files, map[*syntax.File][]byte{file: []byte(src)})
	if !changed {
		t.Fatal("Lower reported no change")
	}
	out, err := ApplyEdits([]byte(src), file.Pos().FileBase(), edits[file])
	if err != nil {
		t.Fatalf("ApplyEdits: %v", err)
	}
	loweredFile := reparseLowered(t, path, out)
	pconf, perrsPtr := checkConfig(imp)
	lowered, _ := pconf.Check("overlaytest", []*syntax.File{loweredFile}, newInfo())
	if err := errors.Join(*perrsPtr...); err != nil {
		t.Fatalf("plain recheck: %v", err)
	}
	return lowered, f.Opts, f.Consts
}

func funcResult(t *testing.T, pkg *types2.Package, name string, i int) types2.Type {
	t.Helper()
	fn, ok := pkg.Scope().Lookup(name).(*types2.Func)
	if !ok {
		t.Fatalf("%s is not a func", name)
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		t.Fatalf("%s has no signature", name)
	}
	return sig.Results().At(i).Type()
}

func funcParam(t *testing.T, pkg *types2.Package, name string, i int) types2.Type {
	t.Helper()
	fn, ok := pkg.Scope().Lookup(name).(*types2.Func)
	if !ok {
		t.Fatalf("%s is not a func", name)
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		t.Fatalf("%s has no signature", name)
	}
	return sig.Params().At(i).Type()
}

func structField(t *testing.T, pkg *types2.Package, typeName, field string) types2.Type {
	t.Helper()
	tn, ok := pkg.Scope().Lookup(typeName).(*types2.TypeName)
	if !ok {
		t.Fatalf("%s is not a type name", typeName)
	}
	st, ok := tn.Type().(*types2.Named).Underlying().(*types2.Struct)
	if !ok {
		t.Fatalf("%s is not a struct", typeName)
	}
	for i := range st.NumFields() {
		if f := st.Field(i); f.Name() == field {
			return f.Type()
		}
	}
	t.Fatalf("%s has no field %s", typeName, field)
	return nil
}

func lookupMethod(t *testing.T, pkg *types2.Package, typeName, method string) *types2.Func {
	t.Helper()
	tn, ok := pkg.Scope().Lookup(typeName).(*types2.TypeName)
	if !ok {
		t.Fatalf("%s is not a type name", typeName)
	}
	named, ok := tn.Type().(*types2.Named)
	if !ok {
		t.Fatalf("%s is not named", typeName)
	}
	for i := range named.NumMethods() {
		if m := named.Method(i); m.Name() == method {
			return m
		}
	}
	t.Fatalf("%s has no method %s", typeName, method)
	return nil
}

func methodParam(t *testing.T, pkg *types2.Package, typeName, method string, i int) types2.Type {
	t.Helper()
	m := lookupMethod(t, pkg, typeName, method)
	sig, ok := m.Type().(*types2.Signature)
	if !ok {
		t.Fatalf("%s.%s has no signature", typeName, method)
	}
	return sig.Params().At(i).Type()
}

func ifaceMethodResult(t *testing.T, pkg *types2.Package, typeName, method string, i int) types2.Type {
	t.Helper()
	tn, ok := pkg.Scope().Lookup(typeName).(*types2.TypeName)
	if !ok {
		t.Fatalf("%s is not a type name", typeName)
	}
	iface, ok := tn.Type().(*types2.Named).Underlying().(*types2.Interface)
	if !ok {
		t.Fatalf("%s is not an interface", typeName)
	}
	for j := range iface.NumExplicitMethods() {
		m := iface.ExplicitMethod(j)
		if m.Name() != method {
			continue
		}
		sig, ok := m.Type().(*types2.Signature)
		if !ok {
			t.Fatalf("%s.%s has no signature", typeName, method)
		}
		return sig.Results().At(i).Type()
	}
	t.Fatalf("%s has no explicit method %s", typeName, method)
	return nil
}
