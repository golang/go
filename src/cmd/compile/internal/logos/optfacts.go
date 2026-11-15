// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/types2"
)

func optAsOption(t types2.Type) *types2.Option {
	if t == nil {
		return nil
	}
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if u := n.Underlying(); u != nil {
			if o, ok := u.(*types2.Option); ok {
				return o
			}
		}
	}
	return nil
}

func joinPath(path []string) string {
	return strings.Join(path, ".")
}

type qualKind uint8

const (
	qualOption qualKind = iota
	qualConst
)

type qualPath struct {
	path string
	qual qualKind
}

func copyFuncReadonly(dst, src *types2.Func) {
	if dst == nil || src == nil {
		return
	}
	if src.LogosReadonly() {
		dst.SetLogosReadonly(true)
	}
}

func copyFuncAssert(dst *types2.Func, assert map[string]facts.FnFact, objects ...string) {
	if dst == nil || len(assert) == 0 {
		return
	}
	for _, o := range objects {
		rec, ok := assert[o]
		if !ok || !rec.HasAssert {
			continue
		}
		tp, ok1 := types2.LogosAssertMaskParse(rec.AssertTP)
		rtp, ok2 := types2.LogosAssertMaskParse(rec.AssertRTP)
		if !ok1 || !ok2 {
			continue
		}
		dst.SetLogosAssertEffects(tp, rtp, true)
		return
	}
}

func ApplyFuncFacts(pkg *types2.Package, fns []facts.FnFact) error {
	if pkg == nil || pkg.Scope() == nil {
		return nil
	}
	objects := reachableFuncFactObjects(pkg)
	type annotation struct {
		fn              *types2.Func
		tp, rtp         []byte
		known, readonly bool
	}
	var pending []annotation
	for _, rec := range fns {
		fn := objects[rec.Object]
		if fn == nil || (!rec.HasAssert && !rec.Readonly) {
			continue
		}
		entry := annotation{fn: fn, readonly: rec.Readonly}
		if rec.HasAssert {
			sig := fn.Signature()
			tp, err := checkedAssertMask(rec.AssertTP, sig.TypeParams().Len())
			if err != nil {
				return fmt.Errorf("logos: assertion facts for %s.%s: AssertTP: %w", pkg.Path(), rec.Object, err)
			}
			rtp, err := checkedAssertMask(rec.AssertRTP, sig.RecvTypeParams().Len())
			if err != nil {
				return fmt.Errorf("logos: assertion facts for %s.%s: AssertRTP: %w", pkg.Path(), rec.Object, err)
			}
			entry.tp, entry.rtp, entry.known = tp, rtp, true
		}
		pending = append(pending, entry)
	}
	for _, entry := range pending {
		if entry.known {
			entry.fn.SetLogosAssertEffects(entry.tp, entry.rtp, true)
		}
		if entry.readonly {
			entry.fn.SetLogosReadonly(true)
		}
	}
	return nil
}

func ApplyReadonlyFactsInPlace(pkg *types2.Package, readonlys []facts.ReadonlyFact) error {
	if pkg == nil || pkg.Scope() == nil {
		return nil
	}
	funcObjs := reachableFuncFactObjects(pkg)
	for _, r := range readonlys {
		if !readonlyPathHasMethod(r.Path) {
			return fmt.Errorf("logos: bad readonly path %q (want at least one M<name> step)", r.Path)
		}
		var base types2.Type
		if obj := pkg.Scope().Lookup(r.Object); obj != nil {
			switch o := obj.(type) {
			case *types2.TypeName:
				base = o.Type()
			case *types2.Func:
				if sig, ok := o.Type().(*types2.Signature); ok {
					base = sig
				}
			}
		}
		if base == nil {
			if fn := funcObjs[r.Object]; fn != nil {
				if sig, ok := fn.Type().(*types2.Signature); ok {
					base = sig
				}
			}
		}
		if base == nil {
			continue
		}
		target, err := readonlyResolveTarget(base, strings.Split(r.Path, "."), make(map[types2.Type]bool))
		if err != nil {
			return fmt.Errorf("logos: readonly %s: %w", r.Object, err)
		}
		if target == nil {
			continue
		}
		if target.Pkg() != pkg {
			continue
		}
		target.SetLogosReadonly(true)
	}
	return nil
}

func checkedAssertMask(mask string, arity int) ([]byte, error) {
	top, ok := facts.HighestAssertBit(mask)
	if !ok {
		return nil, fmt.Errorf("invalid mask %q", mask)
	}
	if top >= arity {
		return nil, fmt.Errorf("mask %q exceeds declaration arity %d", mask, arity)
	}
	parsed, ok := types2.LogosAssertMaskParse(mask)
	if !ok {
		return nil, fmt.Errorf("invalid mask %q", mask)
	}
	return parsed, nil
}

func reachableFuncFactObjects(pkg *types2.Package) map[string]*types2.Func {
	objects := make(map[string]*types2.Func)
	graph := &overlayGraph{orig: pkg, nodes: make(map[types2.Type]bool)}
	refs := make(map[types2.Type]bool)
	seen := make(map[types2.Type]bool)
	visit := func(t types2.Type) { graph.references(t, refs, seen) }
	for _, name := range pkg.Scope().Names() {
		if strings.Contains(name, ".") {
			continue
		}
		obj := pkg.Scope().Lookup(name)
		if fn, ok := obj.(*types2.Func); ok {
			objects[name] = fn.Origin()
		}
		visit(obj.Type())
	}
	visited := make(map[types2.Type]bool)
	for {
		progress := false
		for node := range graph.nodes {
			if visited[node] {
				continue
			}
			visited[node], progress = true, true
			switch n := node.(type) {
			case *types2.Named:
				collectNamedFuncFacts(n, objects, visit)
			case *types2.Alias:
				visit(n.Rhs())
				for i := range n.TypeParams().Len() {
					visit(n.TypeParams().At(i).Constraint())
				}
			}
		}
		if !progress {
			return objects
		}
	}
}

func collectNamedFuncFacts(n *types2.Named, objects map[string]*types2.Func, visit func(types2.Type)) {
	n = n.Origin()
	visit(n.Underlying())
	for i := range n.TypeParams().Len() {
		visit(n.TypeParams().At(i).Constraint())
	}
	for i := range n.NumMethods() {
		m := n.Method(i).Origin()
		objects[facts.MethodObject(n.Obj().Name(), m.Signature().Recv(), m.Name())] = m
		visit(m.Type())
	}
	if iface, ok := n.Underlying().(*types2.Interface); ok {
		for i := range iface.NumExplicitMethods() {
			m := iface.ExplicitMethod(i).Origin()
			objects[n.Obj().Name()+"."+m.Name()] = m
			visit(m.Type())
		}
	}
}

func CloneWithOverlaysAndReadonly(orig *types2.Package, opts, consts []facts.PathFact, fns []facts.FnFact, readonlys []facts.ReadonlyFact, constMaps []facts.ConstMapFact) (*types2.Package, error) {
	var readonly map[string]bool
	var assert map[string]facts.FnFact
	for _, fn := range fns {
		if fn.Object == "" {
			continue
		}
		if fn.Readonly {
			if readonly == nil {
				readonly = make(map[string]bool)
			}
			readonly[fn.Object] = true
		}
		if fn.HasAssert {
			if assert == nil {
				assert = make(map[string]facts.FnFact)
			}
			assert[fn.Object] = fn
		}
	}
	if len(opts) == 0 && len(consts) == 0 && len(constMaps) == 0 {
		if err := ApplyFuncFacts(orig, fns); err != nil {
			return nil, err
		}
		if err := ApplyReadonlyFactsInPlace(orig, readonlys); err != nil {
			return nil, err
		}
		return orig, nil
	}
	byObj := qualByObj(opts, consts)
	byReadonly := make(map[string][]string)
	for _, r := range readonlys {
		if r.Object != "" && r.Path != "" {
			byReadonly[r.Object] = append(byReadonly[r.Object], r.Path)
		}
	}
	if len(byObj) == 0 && len(readonly) == 0 && len(byReadonly) == 0 && len(assert) == 0 && len(constMaps) == 0 {
		return orig, nil
	}
	cloned, err := clonePackageWithOverlays(orig, byObj, readonly, byReadonly, assert, constMaps)
	if err != nil {
		return nil, err
	}
	if err := ApplyFuncFacts(cloned, fns); err != nil {
		return nil, err
	}
	return cloned, nil
}

func applyReadonlyToUnderlying(pkg *types2.Package, owner string, u types2.Type, readonly map[string]bool) types2.Type {
	if len(readonly) == 0 {
		return u
	}
	iface, ok := u.(*types2.Interface)
	if !ok {
		return u
	}
	need := false
	for i := range iface.NumExplicitMethods() {
		m := iface.ExplicitMethod(i)
		if m == nil {
			continue
		}
		if readonly[owner+"."+m.Name()] && !m.LogosReadonly() {
			need = true
			break
		}
	}
	if !need {
		return u
	}
	methods := make([]*types2.Func, iface.NumExplicitMethods())
	for i := range iface.NumExplicitMethods() {
		m := iface.ExplicitMethod(i)
		methods[i] = m
		if m == nil {
			continue
		}
		if !readonly[owner+"."+m.Name()] {
			continue
		}
		if m.LogosReadonly() {
			continue
		}
		sig, ok := m.Type().(*types2.Signature)
		if !ok {
			continue
		}
		nf := types2.NewFunc(m.Pos(), pkg, m.Name(), sig)
		nf.SetLogosReadonly(true)
		methods[i] = nf
	}
	embeddeds := make([]types2.Type, iface.NumEmbeddeds())
	for i := range iface.NumEmbeddeds() {
		embeddeds[i] = iface.EmbeddedType(i)
	}
	return types2.NewInterfaceType(methods, embeddeds)
}

func markOwnersAffected(g *overlayGraph, keys []string) {
	if len(keys) == 0 {
		return
	}
	owners := make(map[string]bool, len(keys))
	for _, key := range keys {
		owners[ownerBase(key)] = true
	}
	for node := range g.nodes {
		var name string
		switch n := node.(type) {
		case *types2.Named:
			name = n.Obj().Name()
		case *types2.Alias:
			name = n.Obj().Name()
		default:
			continue
		}
		if owners[name] {
			g.affected[node] = true
		}
	}
}

func ownerBase(object string) string {
	base := object
	if before, _, ok := strings.CutLast(base, "."); ok {
		base = before
	}
	if rest, ok := strings.CutPrefix(base, "("); ok {
		base = rest
	}
	if rest, ok := strings.CutPrefix(base, "*"); ok {
		base = rest
	}
	if rest, ok := strings.CutSuffix(base, ")"); ok {
		base = rest
	}
	if rest, ok := strings.CutPrefix(base, "*"); ok {
		base = rest
	}
	return base
}

func CloneWithOverlays(orig *types2.Package, opts, consts []facts.PathFact) *types2.Package {
	cloned, err := clonePackageWithOverlays(orig, qualByObj(opts, consts), nil, nil, nil, nil)
	if err != nil {
		panic("overlay qual-only rebuild invariant violated: " + err.Error())
	}
	return cloned
}

func qualByObj(opts, consts []facts.PathFact) map[string][]qualPath {
	byObj := make(map[string][]qualPath)
	for _, o := range opts {
		if o.Object != "" && o.Path != "" {
			byObj[o.Object] = append(byObj[o.Object], qualPath{path: o.Path, qual: qualOption})
		}
	}
	for _, c := range consts {
		if c.Object != "" && c.Path != "" {
			byObj[c.Object] = append(byObj[c.Object], qualPath{path: c.Path, qual: qualConst})
		}
	}
	return byObj
}

func clonePackageWithOverlays(orig *types2.Package, byObj map[string][]qualPath, readonly map[string]bool, byReadonly map[string][]string, assert map[string]facts.FnFact, constMaps []facts.ConstMapFact) (*types2.Package, error) {
	for obj := range byObj {
		paths := byObj[obj]
		slices.SortStableFunc(paths, func(a, b qualPath) int {
			if d := strings.Count(a.path, ".") - strings.Count(b.path, "."); d != 0 {
				return d
			}
			if d := strings.Compare(a.path, b.path); d != 0 {
				return d
			}
			return int(a.qual) - int(b.qual)
		})
	}
	for obj := range byReadonly {
		slices.Sort(byReadonly[obj])
	}
	out := types2.NewPackage(orig.Path(), orig.Name())
	out.SetImports(orig.Imports())
	graph := newOverlayGraph(orig, out, byObj)
	graph.readonly = readonly
	graph.assert = assert
	graph.constMaps = constMapIndex(constMaps)
	for obj, paths := range byReadonly {
		graph.readonlyPaths[obj] = slices.Clone(paths)
	}
	if err := graph.prepare(); err != nil {
		return nil, err
	}
	for _, name := range orig.Scope().Names() {
		if strings.Contains(name, ".") {
			continue
		}
		if err := insertOverlayDecl(out, graph, byObj, readonly, assert, name); err != nil {
			return nil, err
		}
	}
	out.MarkComplete()
	return out, nil
}

func insertOverlayDecl(out *types2.Package, graph *overlayGraph, byObj map[string][]qualPath, readonly map[string]bool, assert map[string]facts.FnFact, name string) error {
	obj := graph.orig.Scope().Lookup(name)
	switch obj := obj.(type) {
	case *types2.TypeName:
		return insertOverlayTypeName(out, graph, obj, name)
	case *types2.Func:
		return insertOverlayFunc(out, graph, byObj, readonly, assert, obj, name)
	case *types2.Const:
		t := substType(graph.sub, obj.Type())
		if t == obj.Type() {
			out.Scope().Insert(obj)
			return nil
		}
		out.Scope().Insert(types2.NewConst(obj.Pos(), out, obj.Name(), t, obj.Val()))
		return nil
	case *types2.Var:
		t := graph.constMapView(name, substType(graph.sub, obj.Type()))
		if t == obj.Type() {
			out.Scope().Insert(obj)
			return nil
		}
		v := types2.NewVar(obj.Pos(), out, obj.Name(), t)
		v.SetKind(obj.Kind())
		out.Scope().Insert(v)
		return nil
	default:
		out.Scope().Insert(obj)
		return nil
	}
}

func insertOverlayTypeName(out *types2.Package, graph *overlayGraph, obj *types2.TypeName, name string) error {
	t := substType(graph.sub, obj.Type())
	if paths := graph.readonlyPaths[name]; len(paths) != 0 {
		var err error
		t, err = applyReadonlyPaths(graph.sub.pkg, graph.orig, t, paths)
		if err != nil {
			return fmt.Errorf("logos: readonly %s: %w", name, err)
		}
	}
	if t == obj.Type() {
		out.Scope().Insert(obj)
		return nil
	}
	switch t := t.(type) {
	case *types2.Named:
		out.Scope().Insert(t.Obj())
	case *types2.Alias:
		out.Scope().Insert(t.Obj())
	default:
		out.Scope().Insert(types2.NewTypeName(obj.Pos(), out, obj.Name(), t))
	}
	return nil
}

func insertOverlayFunc(out *types2.Package, graph *overlayGraph, byObj map[string][]qualPath, readonly map[string]bool, assert map[string]facts.FnFact, obj *types2.Func, name string) error {
	sig, ok := obj.Type().(*types2.Signature)
	if !ok {
		out.Scope().Insert(obj)
		return nil
	}
	sig = substSignature(graph.sub, sig)
	for _, p := range byObj[name] {
		sig = wrapAtQual(out, sig, strings.Split(p.path, "."), p.qual).(*types2.Signature)
	}
	if paths := graph.readonlyPaths[name]; len(paths) != 0 {
		cur, err := applyReadonlyPaths(graph.sub.pkg, graph.orig, sig, paths)
		if err != nil {
			return fmt.Errorf("logos: readonly %s: %w", name, err)
		}
		nsig, ok := cur.(*types2.Signature)
		if !ok {
			return fmt.Errorf("logos: readonly %s: path does not select signature", name)
		}
		sig = nsig
	}
	needReadonly := readonly[name] && !obj.LogosReadonly()
	_, needAssert := assert[name]
	if sig == obj.Type().(*types2.Signature) && !needReadonly && !needAssert {
		out.Scope().Insert(obj)
		return nil
	}
	nf := types2.NewFunc(obj.Pos(), out, obj.Name(), sig)
	copyFuncReadonly(nf, obj)
	copyFuncAssert(nf, assert, name)
	if readonly[name] {
		nf.SetLogosReadonly(true)
	}
	out.Scope().Insert(nf)
	return nil
}

func applyReadonlyPaths(pkg *types2.Package, orig *types2.Package, t types2.Type, paths []string) (types2.Type, error) {
	cur := t
	for _, p := range paths {
		if !readonlyPathHasMethod(p) {
			return nil, fmt.Errorf("logos: bad readonly path %q (want at least one M<name> step)", p)
		}
		var err error
		cur, err = markReadonlyAtPath(pkg, orig, cur, strings.Split(p, "."))
		if err != nil {
			return nil, err
		}
	}
	return cur, nil
}

type overlayGraph struct {
	orig          *types2.Package
	sub           *typeSubst
	paths         map[string][]qualPath
	readonly      map[string]bool
	assert        map[string]facts.FnFact
	readonlyPaths map[string][]string
	constMaps     map[string]facts.ConstMapFact
	nodes         map[types2.Type]bool
	deps          map[types2.Type]map[types2.Type]bool
	affected      map[types2.Type]bool
}

func newOverlayGraph(orig, pkg *types2.Package, paths map[string][]qualPath) *overlayGraph {
	return &overlayGraph{
		orig: orig, paths: paths,
		readonlyPaths: make(map[string][]string),
		sub:           &typeSubst{pkg: pkg, named: make(map[*types2.Named]*types2.Named), aliases: make(map[*types2.Alias]*types2.Alias), tparams: make(map[*types2.TypeParam]*types2.TypeParam)},
		nodes:         make(map[types2.Type]bool), deps: make(map[types2.Type]map[types2.Type]bool), affected: make(map[types2.Type]bool),
	}
}

func (g *overlayGraph) references(t types2.Type, refs map[types2.Type]bool, seen map[types2.Type]bool) {
	if t == nil || seen[t] {
		return
	}
	seen[t] = true
	switch t := t.(type) {
	case *types2.Named:
		if t.Obj().Pkg() == g.orig {
			n := t.Origin()
			refs[n] = true
			g.nodes[n] = true
		}
		for i := range t.TypeArgs().Len() {
			g.references(t.TypeArgs().At(i), refs, seen)
		}
	case *types2.Alias:
		if t.Obj().Pkg() == g.orig {
			a := t.Origin()
			refs[a] = true
			g.nodes[a] = true
		}
		g.references(t.Rhs(), refs, seen)
		for i := range t.TypeArgs().Len() {
			g.references(t.TypeArgs().At(i), refs, seen)
		}
	case *types2.TypeParam:
		refs[t] = true
		g.references(t.Constraint(), refs, seen)
	case *types2.Pointer:
		g.references(t.Elem(), refs, seen)
	case *types2.Chan:
		g.references(t.Elem(), refs, seen)
	case *types2.Slice:
		g.references(t.Elem(), refs, seen)
	case *types2.Array:
		g.references(t.Elem(), refs, seen)
	case *types2.Map:
		g.references(t.Key(), refs, seen)
		g.references(t.Elem(), refs, seen)
	case *types2.Option:
		g.references(t.Elem, refs, seen)
	case *types2.ConstView:
		g.references(t.Elem, refs, seen)
	case *types2.Struct:
		for i := range t.NumFields() {
			g.references(t.Field(i).Type(), refs, seen)
		}
	case *types2.Tuple:
		if t != nil {
			for i := range t.Len() {
				g.references(t.At(i).Type(), refs, seen)
			}
		}
	case *types2.Signature:
		if t.Recv() != nil {
			g.references(t.Recv().Type(), refs, seen)
		}
		g.references(t.Params(), refs, seen)
		g.references(t.Results(), refs, seen)
		for _, list := range []*types2.TypeParamList{t.TypeParams(), t.RecvTypeParams()} {
			for i := range list.Len() {
				g.references(list.At(i).Constraint(), refs, seen)
			}
		}
	case *types2.Interface:
		for i := range t.NumExplicitMethods() {
			g.references(t.ExplicitMethod(i).Type(), refs, seen)
		}
		for i := range t.NumEmbeddeds() {
			g.references(t.EmbeddedType(i), refs, seen)
		}
	case *types2.Union:
		for i := range t.Len() {
			g.references(t.Term(i).Type(), refs, seen)
		}
	}
}

func (g *overlayGraph) route(t types2.Type, steps []string, q qualKind) {
	if t == nil || len(steps) == 0 {
		return
	}
	switch n := t.(type) {
	case *types2.Named:
		if n.Obj().Pkg() != g.orig {
			return
		}
		name := n.Origin().Obj().Name()
		p := qualPath{path: joinPath(steps), qual: q}
		if !slices.Contains(g.paths[name], p) {
			g.paths[name] = append(g.paths[name], p)
		}
		u := n.Underlying()
		for _, outer := range g.paths[name] {
			if strings.HasPrefix(p.path, outer.path+".") {
				u = wrapAtQual(g.sub.pkg, u, strings.Split(outer.path, "."), outer.qual)
			}
		}
		g.route(u, steps, q)
		return
	case *types2.Alias:
		g.route(n.Rhs(), steps, q)
		return
	}
	head, rest := steps[0], steps[1:]
	switch u := t.(type) {
	case *types2.Option:
		if head == "E" {
			g.route(u.Elem, rest, q)
		}
	case *types2.ConstView:
		if head == "E" {
			g.route(u.Elem, rest, q)
		}
	case *types2.Pointer:
		if head == "D" {
			g.route(u.Elem(), rest, q)
		}
	case *types2.Chan:
		if head == "D" {
			g.route(u.Elem(), rest, q)
		}
	case *types2.Slice:
		if head == "E" {
			g.route(u.Elem(), rest, q)
		}
	case *types2.Array:
		if head == "E" {
			g.route(u.Elem(), rest, q)
		}
	case *types2.Map:
		if head == "K" {
			g.route(u.Key(), rest, q)
		}
		if head == "E" {
			g.route(u.Elem(), rest, q)
		}
	case *types2.Struct:
		if len(head) < 2 || head[0] != 'F' {
			return
		}
		for i := range u.NumFields() {
			if f := u.Field(i); f.Name() == head[1:] {
				g.route(f.Type(), rest, q)
				return
			}
		}
	case *types2.Interface:
		if len(head) < 2 || head[0] != 'M' {
			return
		}
		for i := range u.NumExplicitMethods() {
			if m := u.ExplicitMethod(i); m.Name() == head[1:] {
				g.route(m.Type(), rest, q)
				return
			}
		}
	case *types2.Signature:
		if len(head) < 2 {
			return
		}
		var tuple *types2.Tuple
		switch head[0] {
		case 'P':
			tuple = u.Params()
		case 'R':
			tuple = u.Results()
		default:
			return
		}
		var index int
		if _, err := fmt.Sscanf(head[1:], "%d", &index); err != nil {
			return
		}
		if index >= 0 && index < tuple.Len() {
			g.route(tuple.At(index).Type(), rest, q)
		}
	}
}

func (g *overlayGraph) routeObject(object string, t types2.Type) {
	for _, p := range slices.Clone(g.paths[object]) {
		steps := strings.Split(p.path, ".")
		g.route(t, steps, p.qual)
		t = wrapAtQual(g.sub.pkg, t, steps, p.qual)
	}
}

func (g *overlayGraph) routeReadonly(t types2.Type, steps []string, seen map[types2.Type]bool) {
	if t == nil || len(steps) == 0 || seen[t] {
		return
	}
	if _, ok := t.(*types2.Named); ok {
		seen[t] = true
		defer delete(seen, t)
	}
	if _, ok := t.(*types2.Alias); ok {
		seen[t] = true
		defer delete(seen, t)
	}
	if _, ok := t.(*types2.Interface); ok {
		seen[t] = true
		defer delete(seen, t)
	}
	if n, ok := t.(*types2.Named); ok {
		if n.Obj().Pkg() != g.orig {
			return
		}
		name := n.Origin().Obj().Name()
		p := joinPath(steps)
		if !slices.Contains(g.readonlyPaths[name], p) {
			g.readonlyPaths[name] = append(g.readonlyPaths[name], p)
		}
		if len(steps[0]) > 1 && steps[0][0] == 'G' {
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return
			}
			tp := n.TypeParams()
			if tp == nil || idx < 0 || idx >= tp.Len() {
				return
			}
			g.routeReadonly(tp.At(idx).Constraint(), steps[1:], seen)
			return
		}
		g.routeReadonly(n.Underlying(), steps, seen)
		return
	}
	if a, ok := t.(*types2.Alias); ok {
		if len(steps[0]) > 1 && steps[0][0] == 'G' {
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return
			}
			tp := a.TypeParams()
			if tp == nil || idx < 0 || idx >= tp.Len() {
				return
			}
			g.routeReadonly(tp.At(idx).Constraint(), steps[1:], seen)
			return
		}
		g.routeReadonly(a.Rhs(), steps, seen)
		return
	}
	if p, ok := t.(*types2.TypeParam); ok {
		g.routeReadonly(p.Constraint(), steps, seen)
		return
	}
	head, rest := steps[0], steps[1:]
	if o := optAsOption(t); o != nil {
		if head == "E" {
			g.routeReadonly(o.Elem, rest, seen)
		}
		return
	}
	if c := constAsView(t); c != nil {
		if head == "E" {
			g.routeReadonly(c.Elem, rest, seen)
		}
		return
	}
	switch u := types2.Unalias(t).(type) {
	case *types2.Pointer:
		if head == "D" {
			g.routeReadonly(u.Elem(), rest, seen)
		}
	case *types2.Chan:
		if head == "D" {
			g.routeReadonly(u.Elem(), rest, seen)
		}
	case *types2.Slice:
		if head == "E" {
			g.routeReadonly(u.Elem(), rest, seen)
		}
	case *types2.Array:
		if head == "E" {
			g.routeReadonly(u.Elem(), rest, seen)
		}
	case *types2.Map:
		if head == "K" {
			g.routeReadonly(u.Key(), rest, seen)
		}
		if head == "E" {
			g.routeReadonly(u.Elem(), rest, seen)
		}
	case *types2.Struct:
		if len(head) < 2 || head[0] != 'F' {
			return
		}
		for i := range u.NumFields() {
			if f := u.Field(i); f.Name() == head[1:] {
				g.routeReadonly(f.Type(), rest, seen)
				return
			}
		}
	case *types2.Interface:
		if len(head) >= 2 && head[0] == 'M' {
			for i := range u.NumExplicitMethods() {
				if m := u.ExplicitMethod(i); m.Name() == head[1:] {
					g.routeReadonly(m.Type(), rest, seen)
					return
				}
			}
			for i := range u.NumEmbeddeds() {
				g.routeReadonly(u.EmbeddedType(i), steps, seen)
			}
			return
		}
		if head == "E" {
			return
		}
	case *types2.Signature:
		if len(head) >= 2 {
			switch head[0] {
			case 'P', 'R':
				idx, ok := parsePathIndex(head[1:])
				if !ok {
					return
				}
				var tup *types2.Tuple
				if head[0] == 'P' {
					tup = u.Params()
				} else {
					tup = u.Results()
				}
				if tup != nil && idx >= 0 && idx < tup.Len() {
					g.routeReadonly(tup.At(idx).Type(), rest, seen)
				}
				return
			case 'G', 'H':
				idx, ok := parsePathIndex(head[1:])
				if !ok {
					return
				}
				var list *types2.TypeParamList
				if head[0] == 'G' {
					list = u.TypeParams()
				} else {
					list = u.RecvTypeParams()
				}
				if list != nil && idx >= 0 && idx < list.Len() {
					g.routeReadonly(list.At(idx).Constraint(), rest, seen)
				}
				return
			}
		}
	case *types2.Tuple:
		idx, ok := parsePathIndex(head[1:])
		if !ok || len(head) < 2 || head[0] != 'T' {
			return
		}
		if idx >= 0 && idx < u.Len() {
			g.routeReadonly(u.At(idx).Type(), rest, seen)
		}
	case *types2.Union:
		for i := range u.Len() {
			g.routeReadonly(u.Term(i).Type(), steps, seen)
		}
	}
}

func (g *overlayGraph) routeReadonlyObject(object string, t types2.Type) {
	for _, p := range slices.Clone(g.readonlyPaths[object]) {
		g.routeReadonly(t, strings.Split(p, "."), make(map[types2.Type]bool))
	}
}

func parsePathIndex(s string) (int, bool) {
	if s == "" {
		return 0, false
	}
	v := 0
	for i := range s {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
		v = v*10 + int(s[i]-'0')
	}
	return v, true
}

func readonlyPathHasMethod(path string) bool {
	for step := range strings.SplitSeq(path, ".") {
		if len(step) > 1 && step[0] == 'M' {
			return true
		}
	}
	return false
}

func checkNominalDefer(pkg *types2.Package, orig *types2.Package, t types2.Type, steps []string) (types2.Type, error) {
	if _, ok := t.(*types2.Alias); ok {
		return checkNominalTarget(pkg, orig, t, steps)
	}
	if n, ok := t.(*types2.Named); ok {
		if n.Obj().Pkg() == pkg {
			return t, nil
		}
		return checkNominalTarget(pkg, orig, t, steps)
	}
	return t, nil
}

func checkNominalTarget(pkg *types2.Package, orig *types2.Package, t types2.Type, steps []string) (types2.Type, error) {
	target, err := readonlyResolveTarget(t, steps, make(map[types2.Type]bool))
	if err != nil {
		return nil, err
	}
	if target == nil {
		return nil, fmt.Errorf("logos: bad readonly path %q (unknown method)", joinPath(steps))
	}
	if target.LogosReadonly() {
		return t, nil
	}
	if isLocalNominal(t, orig, pkg) {
		return t, nil
	}
	return nil, fmt.Errorf("logos: bad readonly path %q (foreign method without readonly obligation)", joinPath(steps))
}

func isLocalNominal(t types2.Type, orig *types2.Package, pkg *types2.Package) bool {
	var objPkg *types2.Package
	switch n := t.(type) {
	case *types2.Named:
		if n.Obj() == nil {
			return false
		}
		objPkg = n.Obj().Pkg()
	case *types2.Alias:
		if n.Obj() == nil {
			return false
		}
		objPkg = n.Obj().Pkg()
	default:
		return false
	}
	return objPkg == orig || objPkg == pkg
}

func readonlyResolveTarget(t types2.Type, steps []string, seen map[types2.Type]bool) (*types2.Func, error) {
	if t == nil || len(steps) == 0 || seen[t] {
		return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
	}
	seen[t] = true
	defer delete(seen, t)
	if a, ok := t.(*types2.Alias); ok {
		if len(steps[0]) > 1 && steps[0][0] == 'G' {
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			tp := a.TypeParams()
			if tp == nil || idx < 0 || idx >= tp.Len() {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			return readonlyResolveTarget(tp.At(idx).Constraint(), steps[1:], seen)
		}
		return readonlyResolveTarget(a.Rhs(), steps, seen)
	}
	if n, ok := t.(*types2.Named); ok {
		if len(steps[0]) > 1 && steps[0][0] == 'G' {
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			if tp := n.TypeParams(); tp != nil && idx >= 0 && idx < tp.Len() {
				return readonlyResolveTarget(tp.At(idx).Constraint(), steps[1:], seen)
			}
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(n.Underlying(), steps, seen)
	}
	if o := optAsOption(t); o != nil {
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(o.Elem, steps[1:], seen)
	}
	if c := constAsView(t); c != nil {
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(c.Elem, steps[1:], seen)
	}
	if p, ok := t.(*types2.TypeParam); ok {
		return readonlyResolveTarget(p.Constraint(), steps, seen)
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Pointer:
		if steps[0] != "D" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(u.Elem(), steps[1:], seen)
	case *types2.Chan:
		if steps[0] != "D" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(u.Elem(), steps[1:], seen)
	case *types2.Slice:
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(u.Elem(), steps[1:], seen)
	case *types2.Array:
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return readonlyResolveTarget(u.Elem(), steps[1:], seen)
	case *types2.Map:
		if steps[0] == "K" {
			return readonlyResolveTarget(u.Key(), steps[1:], seen)
		}
		if steps[0] == "E" {
			return readonlyResolveTarget(u.Elem(), steps[1:], seen)
		}
		return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
	case *types2.Struct:
		if len(steps[0]) < 2 || steps[0][0] != 'F' {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		for i := range u.NumFields() {
			if f := u.Field(i); f != nil && f.Name() == steps[0][1:] {
				return readonlyResolveTarget(f.Type(), steps[1:], seen)
			}
		}
		return nil, fmt.Errorf("logos: bad readonly path %q (unknown field)", joinPath(steps))
	case *types2.Signature:
		if len(steps[0]) < 2 {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		switch steps[0][0] {
		case 'P', 'R':
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			var tup *types2.Tuple
			if steps[0][0] == 'P' {
				tup = u.Params()
			} else {
				tup = u.Results()
			}
			if tup == nil || idx < 0 || idx >= tup.Len() {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			return readonlyResolveTarget(tup.At(idx).Type(), steps[1:], seen)
		case 'G', 'H':
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			var list *types2.TypeParamList
			if steps[0][0] == 'G' {
				list = u.TypeParams()
			} else {
				list = u.RecvTypeParams()
			}
			if list == nil || idx < 0 || idx >= list.Len() {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			return readonlyResolveTarget(list.At(idx).Constraint(), steps[1:], seen)
		default:
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
	case *types2.Interface:
		if len(steps[0]) < 2 || steps[0][0] != 'M' {
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		name := steps[0][1:]
		for i := range u.NumExplicitMethods() {
			if m := u.ExplicitMethod(i); m != nil && m.Name() == name {
				if len(steps) == 1 {
					return m, nil
				}
				return readonlyResolveTarget(m.Type(), steps[1:], seen)
			}
		}
		for i := range u.NumEmbeddeds() {
			if m, err := readonlyResolveTarget(u.EmbeddedType(i), steps, seen); err == nil && m != nil {
				return m, nil
			}
		}
		return nil, fmt.Errorf("logos: bad readonly path %q (unknown method)", joinPath(steps))
	default:
		if n, ok := t.(*types2.Named); ok && len(steps) > 0 && len(steps[0]) > 1 && steps[0][0] == 'G' {
			idx, ok := parsePathIndex(steps[0][1:])
			if !ok {
				return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
			}
			if tp := n.TypeParams(); tp != nil && idx >= 0 && idx < tp.Len() {
				return readonlyResolveTarget(tp.At(idx).Constraint(), steps[1:], seen)
			}
			return nil, fmt.Errorf("logos: bad readonly path %q", joinPath(steps))
		}
		return nil, fmt.Errorf("logos: bad readonly path %q (cannot descend into %T)", joinPath(steps), t)
	}
}

func markReadonlyAtPath(pkg *types2.Package, orig *types2.Package, t types2.Type, steps []string) (types2.Type, error) {
	if len(steps) == 0 {
		return nil, fmt.Errorf("logos: bad readonly path (empty steps)")
	}
	if t == nil {
		return nil, fmt.Errorf("logos: bad readonly path %q (nil type)", joinPath(steps))
	}
	if _, ok := t.(*types2.Alias); ok {
		return checkNominalDefer(pkg, orig, t, steps)
	}
	if _, ok := t.(*types2.Named); ok {
		return checkNominalDefer(pkg, orig, t, steps)
	}
	if o := optAsOption(t); o != nil {
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want E through option)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, o.Elem, steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == o.Elem {
			return t, nil
		}
		return &types2.Option{Elem: elem}, nil
	}
	if c := constAsView(t); c != nil {
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want E through const view)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, c.Elem, steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == c.Elem {
			return t, nil
		}
		return &types2.ConstView{Elem: elem, ConstMap: c.ConstMap}, nil
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Pointer:
		if steps[0] != "D" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want D through pointer)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, u.Elem(), steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == u.Elem() {
			return t, nil
		}
		return types2.NewPointer(elem), nil
	case *types2.Chan:
		if steps[0] != "D" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want D through channel)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, u.Elem(), steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == u.Elem() {
			return t, nil
		}
		return types2.NewChan(u.Dir(), elem), nil
	case *types2.Slice:
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want E through slice)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, u.Elem(), steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == u.Elem() {
			return t, nil
		}
		return types2.NewSlice(elem), nil
	case *types2.Array:
		if steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want E through array)", joinPath(steps))
		}
		elem, err := markReadonlyAtPath(pkg, orig, u.Elem(), steps[1:])
		if err != nil {
			return nil, err
		}
		if elem == u.Elem() {
			return t, nil
		}
		return types2.NewArray(elem, u.Len()), nil
	case *types2.Map:
		if steps[0] != "K" && steps[0] != "E" {
			return nil, fmt.Errorf("logos: bad readonly path %q (want K or E through map)", joinPath(steps))
		}
		if steps[0] == "K" {
			k, err := markReadonlyAtPath(pkg, orig, u.Key(), steps[1:])
			if err != nil {
				return nil, err
			}
			if k == u.Key() {
				return t, nil
			}
			return types2.NewMap(k, u.Elem()), nil
		}
		e, err := markReadonlyAtPath(pkg, orig, u.Elem(), steps[1:])
		if err != nil {
			return nil, err
		}
		if e == u.Elem() {
			return t, nil
		}
		return types2.NewMap(u.Key(), e), nil
	case *types2.Struct:
		if len(steps[0]) < 2 || steps[0][0] != 'F' {
			return nil, fmt.Errorf("logos: bad readonly path %q (want F<name> through struct)", joinPath(steps))
		}
		return markReadonlyStructField(pkg, orig, u, steps[0][1:], steps[1:])
	case *types2.Signature:
		return markReadonlySig(pkg, orig, u, steps)
	case *types2.Interface:
		return markReadonlyInterface(pkg, orig, u, steps)
	case *types2.Tuple:
		if len(steps[0]) < 2 || steps[0][0] != 'T' {
			return nil, fmt.Errorf("logos: bad readonly path %q (want T<i> through tuple)", joinPath(steps))
		}
		idx, ok := parsePathIndex(steps[0][1:])
		if !ok || idx < 0 || idx >= u.Len() {
			return nil, fmt.Errorf("logos: bad readonly path %q (bad tuple index)", joinPath(steps))
		}
		vars := make([]*types2.Var, u.Len())
		for i := range u.Len() {
			vars[i] = u.At(i)
		}
		nt, err := markReadonlyAtPath(pkg, orig, vars[idx].Type(), steps[1:])
		if err != nil {
			return nil, err
		}
		if nt == vars[idx].Type() {
			return t, nil
		}
		nv := types2.NewVar(vars[idx].Pos(), pkg, vars[idx].Name(), nt)
		nv.SetKind(vars[idx].Kind())
		vars[idx] = nv
		return types2.NewTuple(vars...), nil
	case *types2.Union:
		var terms []*types2.Term
		changed := false
		var firstErr error
		deferred := false
		for i := range u.Len() {
			tm := u.Term(i)
			nt, err := markReadonlyAtPath(pkg, orig, tm.Type(), steps)
			if err != nil {
				if firstErr == nil {
					firstErr = err
				}
				continue
			}
			if nt != tm.Type() {
				if terms == nil {
					terms = make([]*types2.Term, u.Len())
					for j := range u.Len() {
						terms[j] = u.Term(j)
					}
				}
				terms[i] = types2.NewTerm(tm.Tilde(), nt)
				changed = true
			} else {
				if _, ok := types2.Unalias(tm.Type()).(*types2.Named); ok {
					deferred = true
				}
				if _, ok := tm.Type().(*types2.Alias); ok {
					deferred = true
				}
			}
		}
		if changed {
			return types2.NewUnion(terms), nil
		}
		if deferred {
			return t, nil
		}
		if firstErr != nil {
			return nil, firstErr
		}
		return nil, fmt.Errorf("logos: bad readonly path %q (no union term matches)", joinPath(steps))
	default:
		return nil, fmt.Errorf("logos: bad readonly path %q (cannot descend into %T)", joinPath(steps), t)
	}
}

func markReadonlyStructField(pkg *types2.Package, orig *types2.Package, u *types2.Struct, name string, rest []string) (types2.Type, error) {
	idx := -1
	for i := range u.NumFields() {
		if f := u.Field(i); f != nil && f.Name() == name {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil, fmt.Errorf("logos: bad readonly path (unknown field %q)", name)
	}
	nt, err := markReadonlyAtPath(pkg, orig, u.Field(idx).Type(), rest)
	if err != nil {
		return nil, err
	}
	if nt == u.Field(idx).Type() {
		return u, nil
	}
	fields := make([]*types2.Var, u.NumFields())
	tags := make([]string, u.NumFields())
	for i := range u.NumFields() {
		f := u.Field(i)
		tags[i] = u.Tag(i)
		if i != idx {
			fields[i] = f
			continue
		}
		nf := types2.NewField(f.Pos(), pkg, f.Name(), nt, f.Embedded())
		fields[i] = nf
	}
	return types2.NewStruct(fields, tags), nil
}

func markReadonlySig(pkg *types2.Package, orig *types2.Package, sig *types2.Signature, steps []string) (*types2.Signature, error) {
	if len(steps) == 0 {
		return nil, fmt.Errorf("logos: bad readonly path (empty signature steps)")
	}
	head, rest := steps[0], steps[1:]
	if len(head) >= 2 && (head[0] == 'G' || head[0] == 'H') {
		idx, ok := parsePathIndex(head[1:])
		if !ok {
			return nil, fmt.Errorf("logos: bad readonly path %q (bad type parameter index)", joinPath(steps))
		}
		var list *types2.TypeParamList
		if head[0] == 'G' {
			list = sig.TypeParams()
		} else {
			list = sig.RecvTypeParams()
		}
		if list == nil || idx < 0 || idx >= list.Len() {
			return nil, fmt.Errorf("logos: bad readonly path %q (type parameter index out of range)", joinPath(steps))
		}
		inner, newT, newR := freshSigTypeParams(pkg, sig)
		var target *types2.TypeParam
		if head[0] == 'G' {
			if newT == nil {
				return nil, fmt.Errorf("logos: bad readonly path %q (no type parameters)", joinPath(steps))
			}
			target = newT[idx]
		} else {
			if newR == nil {
				return nil, fmt.Errorf("logos: bad readonly path %q (no receiver type parameters)", joinPath(steps))
			}
			target = newR[idx]
		}
		nt, err := markReadonlyAtPath(pkg, orig, target.Constraint(), rest)
		if err != nil {
			return nil, err
		}
		if nt != target.Constraint() {
			target.SetConstraint(nt)
		}
		_ = inner
		return rebuildSigWithFresh(pkg, sig, newT, newR, inner), nil
	}
	if len(head) < 2 || (head[0] != 'P' && head[0] != 'R') {
		return nil, fmt.Errorf("logos: bad readonly path %q (want P<i>, R<i>, G<i> or H<i> through signature)", joinPath(steps))
	}
	idx, ok := parsePathIndex(head[1:])
	if !ok {
		return nil, fmt.Errorf("logos: bad readonly path %q (bad signature index)", joinPath(steps))
	}
	inner, newT, newR := freshSigTypeParams(pkg, sig)
	var tup *types2.Tuple
	if head[0] == 'P' {
		tup = substTupleFresh(inner, sig.Params())
	} else {
		tup = substTupleFresh(inner, sig.Results())
	}
	_ = newT
	_ = newR
	if tup == nil || idx < 0 || idx >= tup.Len() {
		return nil, fmt.Errorf("logos: bad readonly path %q (signature index out of range)", joinPath(steps))
	}
	nt, err := markReadonlyAtPath(pkg, orig, tup.At(idx).Type(), rest)
	if err != nil {
		return nil, err
	}
	if nt == tup.At(idx).Type() {
		return rebuildSigWithFresh(pkg, sig, newT, newR, inner), nil
	}
	vars := make([]*types2.Var, tup.Len())
	for i := range tup.Len() {
		vars[i] = tup.At(i)
	}
	nv := types2.NewVar(vars[idx].Pos(), pkg, vars[idx].Name(), nt)
	nv.SetKind(vars[idx].Kind())
	vars[idx] = nv
	ntup := types2.NewTuple(vars...)
	var params, results *types2.Tuple
	if head[0] == 'P' {
		params, results = ntup, substTupleFresh(inner, sig.Results())
	} else {
		params, results = substTupleFresh(inner, sig.Params()), ntup
	}
	return types2.NewSignatureType(sig.Recv(), newR, newT, params, results, sig.Variadic()), nil
}

func freshSigTypeParams(pkg *types2.Package, sig *types2.Signature) (*typeSubst, []*types2.TypeParam, []*types2.TypeParam) {
	base := &typeSubst{pkg: pkg}
	inner, newT := freshOwnTypeParams(base, sig.TypeParams())
	inner, newR := freshOwnTypeParams(inner, sig.RecvTypeParams())
	return inner, newT, newR
}

func substTupleFresh(sub *typeSubst, tup *types2.Tuple) *types2.Tuple {
	if tup == nil {
		return nil
	}
	vars := make([]*types2.Var, tup.Len())
	for i := range tup.Len() {
		v := tup.At(i)
		e := substType(sub, v.Type())
		if e == v.Type() {
			vars[i] = v
			continue
		}
		nv := types2.NewVar(v.Pos(), sub.pkg, v.Name(), e)
		nv.SetKind(v.Kind())
		vars[i] = nv
	}
	return types2.NewTuple(vars...)
}

func rebuildSigWithFresh(pkg *types2.Package, sig *types2.Signature, newT, newR []*types2.TypeParam, inner *typeSubst) *types2.Signature {
	params := substTupleFresh(inner, sig.Params())
	results := substTupleFresh(inner, sig.Results())
	var nrecv *types2.Var
	if r := sig.Recv(); r != nil {
		rt := substType(inner, r.Type())
		nrecv = types2.NewVar(r.Pos(), pkg, r.Name(), rt)
		nrecv.SetKind(r.Kind())
	}
	if newT == nil && newR == nil && params == sig.Params() && results == sig.Results() && (sig.Recv() == nil || nrecv.Type() == sig.Recv().Type()) {
		return sig
	}
	return types2.NewSignatureType(nrecv, newR, newT, params, results, sig.Variadic())
}

func markReadonlyInterface(pkg *types2.Package, orig *types2.Package, iface *types2.Interface, steps []string) (types2.Type, error) {
	if len(steps) == 0 {
		return nil, fmt.Errorf("logos: bad readonly path (empty interface steps)")
	}
	head, rest := steps[0], steps[1:]
	if len(head) < 2 || head[0] != 'M' {
		return nil, fmt.Errorf("logos: bad readonly path %q (want M<name> through interface)", joinPath(steps))
	}
	name := head[1:]
	for i := range iface.NumExplicitMethods() {
		m := iface.ExplicitMethod(i)
		if m == nil || m.Name() != name {
			continue
		}
		sig, ok := m.Type().(*types2.Signature)
		if !ok {
			return nil, fmt.Errorf("logos: bad readonly path %q (method %q not a signature)", joinPath(steps), name)
		}
		methods := make([]*types2.Func, iface.NumExplicitMethods())
		for j := range iface.NumExplicitMethods() {
			methods[j] = iface.ExplicitMethod(j)
		}
		if len(rest) == 0 {
			if m.LogosReadonly() {
				return iface, nil
			}
			nf := types2.NewFunc(m.Pos(), pkg, m.Name(), sig)
			nf.SetLogosReadonly(true)
			methods[i] = nf
			embeddeds := make([]types2.Type, iface.NumEmbeddeds())
			for j := range iface.NumEmbeddeds() {
				embeddeds[j] = iface.EmbeddedType(j)
			}
			return types2.NewInterfaceType(methods, embeddeds), nil
		}
		nsig, err := markReadonlySig(pkg, orig, sig, rest)
		if err != nil {
			return nil, err
		}
		if nsig == sig {
			return iface, nil
		}
		nf := types2.NewFunc(m.Pos(), pkg, m.Name(), nsig)
		copyFuncReadonly(nf, m)
		methods[i] = nf
		embeddeds := make([]types2.Type, iface.NumEmbeddeds())
		for j := range iface.NumEmbeddeds() {
			embeddeds[j] = iface.EmbeddedType(j)
		}
		return types2.NewInterfaceType(methods, embeddeds), nil
	}
	for i := range iface.NumEmbeddeds() {
		emb := iface.EmbeddedType(i)
		if ifaceHasMethod(emb, name, make(map[types2.Type]bool)) {
			nt, err := markReadonlyAtPath(pkg, orig, emb, steps)
			if err != nil {
				return nil, err
			}
			if nt == emb {
				return iface, nil
			}
			embeddeds := make([]types2.Type, iface.NumEmbeddeds())
			for j := range iface.NumEmbeddeds() {
				if j == i {
					embeddeds[j] = nt
				} else {
					embeddeds[j] = iface.EmbeddedType(j)
				}
			}
			methods := make([]*types2.Func, iface.NumExplicitMethods())
			for j := range iface.NumExplicitMethods() {
				methods[j] = iface.ExplicitMethod(j)
			}
			return types2.NewInterfaceType(methods, embeddeds), nil
		}
	}
	return nil, fmt.Errorf("logos: bad readonly path %q (unknown method %q)", joinPath(steps), name)
}

func ifaceHasMethod(t types2.Type, name string, seen map[types2.Type]bool) bool {
	if t == nil || seen[t] {
		return false
	}
	seen[t] = true
	if a, ok := t.(*types2.Alias); ok {
		return ifaceHasMethod(a.Rhs(), name, seen)
	}
	if n, ok := t.(*types2.Named); ok {
		if u, ok := n.Underlying().(*types2.Interface); ok {
			return ifaceHasMethod(u, name, seen)
		}
		return false
	}
	u, ok := types2.Unalias(t).(*types2.Interface)
	if !ok {
		return false
	}
	for i := range u.NumExplicitMethods() {
		if m := u.ExplicitMethod(i); m != nil && m.Name() == name {
			return true
		}
	}
	for i := range u.NumEmbeddeds() {
		if ifaceHasMethod(u.EmbeddedType(i), name, seen) {
			return true
		}
	}
	return false
}

func (g *overlayGraph) prepare() error {
	for _, name := range g.orig.Scope().Names() {
		if strings.Contains(name, ".") {
			continue
		}
		obj := g.orig.Scope().Lookup(name)
		g.routeObject(name, obj.Type())
		g.routeReadonlyObject(name, obj.Type())
		if n, ok := obj.Type().(*types2.Named); ok {
			for i := range n.NumMethods() {
				m := n.Method(i)
				key := facts.MethodObject(name, m.Type().(*types2.Signature).Recv(), m.Name())
				g.routeObject(key, m.Type())
				g.routeReadonlyObject(key, m.Type())
			}
		}
	}
	for name, paths := range g.readonlyPaths {
		slices.Sort(paths)
		g.readonlyPaths[name] = paths
	}
	for _, name := range g.orig.Scope().Names() {
		if strings.Contains(name, ".") {
			continue
		}
		g.references(g.orig.Scope().Lookup(name).Type(), make(map[types2.Type]bool), make(map[types2.Type]bool))
	}
	for {
		changed := false
		for node := range g.nodes {
			if _, ok := g.deps[node]; ok {
				continue
			}
			changed = true
			refs := make(map[types2.Type]bool)
			g.deps[node] = refs
			seen := make(map[types2.Type]bool)
			var name string
			switch n := node.(type) {
			case *types2.Named:
				name = n.Obj().Name()
				g.references(n.Underlying(), refs, seen)
				for i := range n.NumMethods() {
					m := n.Method(i)
					g.references(m.Type(), refs, seen)
					key := facts.MethodObject(name, m.Type().(*types2.Signature).Recv(), m.Name())
					if len(g.paths[key]) != 0 || g.readonly[key] || len(g.readonlyPaths[key]) != 0 {
						g.affected[node] = true
					}
				}
				for i := range n.TypeParams().Len() {
					g.references(n.TypeParams().At(i).Constraint(), refs, seen)
				}
			case *types2.Alias:
				name = n.Obj().Name()
				g.references(n.Rhs(), refs, seen)
				for i := range n.TypeParams().Len() {
					g.references(n.TypeParams().At(i).Constraint(), refs, seen)
				}
			}
			if len(g.paths[name]) != 0 || g.readonly[name] || len(g.readonlyPaths[name]) != 0 {
				g.affected[node] = true
			}
		}
		if !changed {
			break
		}
	}
	markOwnersAffected(g, slices.Sorted(maps.Keys(g.readonly)))
	markReadonlyPathsAffected(g)
	markOwnersAffected(g, slices.Sorted(maps.Keys(g.assert)))
	for {
		changed := false
		for node, refs := range g.deps {
			if g.affected[node] {
				continue
			}
			for ref := range refs {
				if g.affected[ref] {
					g.affected[node] = true
					changed = true
					break
				}
			}
		}
		if !changed {
			break
		}
	}
	for name, paths := range g.paths {
		slices.SortStableFunc(paths, func(a, b qualPath) int {
			if d := strings.Count(a.path, ".") - strings.Count(b.path, "."); d != 0 {
				return d
			}
			return strings.Compare(a.path, b.path)
		})
		g.paths[name] = paths
	}
	for name, paths := range g.readonlyPaths {
		slices.Sort(paths)
		g.readonlyPaths[name] = paths
	}
	for node := range g.nodes {
		var name string
		switch n := node.(type) {
		case *types2.Named:
			name = n.Obj().Name()
		case *types2.Alias:
			name = n.Obj().Name()
		default:
			continue
		}
		if len(g.readonlyPaths[name]) != 0 {
			g.affected[node] = true
		}
	}
	for node := range g.affected {
		if n, ok := node.(*types2.Named); ok {
			obj := n.Obj()
			g.sub.named[n] = types2.NewNamed(types2.NewTypeName(obj.Pos(), g.sub.pkg, obj.Name(), nil), nil, nil)
		}
	}
	for old, shell := range g.sub.named {
		params := typeParamSlice(old.TypeParams())
		news := make([]*types2.TypeParam, len(params))
		for i, p := range params {
			news[i] = types2.NewTypeParam(types2.NewTypeName(p.Obj().Pos(), g.sub.pkg, p.Obj().Name(), nil), nil)
			g.sub.tparams[p] = news[i]
		}
		shell.SetTypeParams(news)
	}
	g.sub.graph = g
	for node := range g.affected {
		if a, ok := node.(*types2.Alias); ok {
			g.alias(a)
		}
	}
	for old := range g.sub.named {
		for i := range old.TypeParams().Len() {
			p := old.TypeParams().At(i)
			g.sub.tparams[p].SetConstraint(substType(g.sub, p.Constraint()))
		}
	}
	for node := range g.affected {
		if a, ok := node.(*types2.Alias); ok {
			if err := g.applyAliasReadonly(a); err != nil {
				return err
			}
		}
	}
	for old, shell := range g.sub.named {
		if err := g.fillNamedShell(old, shell); err != nil {
			return err
		}
	}
	for old, shell := range g.sub.named {
		for i := range old.NumMethods() {
			if err := g.addShellMethod(old, shell, i); err != nil {
				return err
			}
		}
	}
	return nil
}

func markReadonlyPathsAffected(g *overlayGraph) {
	owners := make(map[string]bool)
	for key := range g.readonlyPaths {
		owners[key] = true
		owners[ownerBase(key)] = true
	}
	for node := range g.nodes {
		var name string
		switch n := node.(type) {
		case *types2.Named:
			name = n.Obj().Name()
		case *types2.Alias:
			name = n.Obj().Name()
		default:
			continue
		}
		if owners[name] {
			g.affected[node] = true
		}
	}
}

func (g *overlayGraph) fillNamedShell(old *types2.Named, shell *types2.Named) error {
	u := substType(g.sub, old.Underlying())
	for _, p := range g.paths[old.Obj().Name()] {
		u = wrapAtQual(g.sub.pkg, u, strings.Split(p.path, "."), p.qual)
	}
	u = applyReadonlyToUnderlying(g.sub.pkg, old.Obj().Name(), u, g.readonly)
	structPaths, genericPaths := splitReadonlyPaths(g.readonlyPaths[old.Obj().Name()])
	var err error
	u, err = applyReadonlyPaths(g.sub.pkg, g.orig, u, structPaths)
	if err != nil {
		return err
	}
	if err := markShellTypeParams(g.sub.pkg, g.orig, shell, genericPaths); err != nil {
		return err
	}
	shell.SetUnderlying(u)
	return nil
}

func splitReadonlyPaths(paths []string) (structPaths []string, genericPaths []string) {
	for _, p := range paths {
		head, _, _ := strings.Cut(p, ".")
		if len(head) > 1 && (head[0] == 'G' || head[0] == 'H') {
			genericPaths = append(genericPaths, p)
			continue
		}
		structPaths = append(structPaths, p)
	}
	return structPaths, genericPaths
}

func markShellTypeParams(pkg *types2.Package, orig *types2.Package, shell *types2.Named, paths []string) error {
	if len(paths) == 0 {
		return nil
	}
	tp := shell.TypeParams()
	if tp == nil {
		return fmt.Errorf("logos: bad readonly path (no type parameters on %s)", shell.Obj().Name())
	}
	for _, p := range paths {
		steps := strings.Split(p, ".")
		head := steps[0]
		if len(head) < 2 || (head[0] != 'G' && head[0] != 'H') {
			return fmt.Errorf("logos: bad readonly path %q (want G<i> through type)", p)
		}
		idx, ok := parsePathIndex(head[1:])
		if !ok || idx < 0 || idx >= tp.Len() {
			return fmt.Errorf("logos: bad readonly path %q (bad type parameter index)", p)
		}
		at := tp.At(idx)
		nt, err := markReadonlyAtPath(pkg, orig, at.Constraint(), steps[1:])
		if err != nil {
			return err
		}
		if nt != at.Constraint() {
			at.SetConstraint(nt)
		}
	}
	return nil
}

func (g *overlayGraph) addShellMethod(old *types2.Named, shell *types2.Named, i int) error {
	m := old.Method(i)
	sig, ok := m.Type().(*types2.Signature)
	if !ok {
		shell.AddMethod(m)
		return nil
	}
	sig = substSignature(g.sub, sig)
	key := facts.MethodObject(old.Obj().Name(), m.Type().(*types2.Signature).Recv(), m.Name())
	for _, p := range g.paths[key] {
		sig = wrapAtQual(g.sub.pkg, sig, strings.Split(p.path, "."), p.qual).(*types2.Signature)
	}
	if paths := g.readonlyPaths[key]; len(paths) != 0 {
		cur, err := applyReadonlyPaths(g.sub.pkg, g.orig, sig, paths)
		if err != nil {
			return fmt.Errorf("logos: readonly %s: %w", key, err)
		}
		nsig, ok := cur.(*types2.Signature)
		if !ok {
			return fmt.Errorf("logos: readonly %s: path does not select signature", key)
		}
		sig = nsig
	}
	nf := types2.NewFunc(m.Pos(), g.sub.pkg, m.Name(), sig)
	copyFuncReadonly(nf, m)
	copyFuncAssert(nf, g.assert, key)
	if g.readonly[key] {
		nf.SetLogosReadonly(true)
	}
	shell.AddMethod(nf)
	return nil
}

func (g *overlayGraph) alias(a *types2.Alias) *types2.Alias {
	if n := g.sub.aliases[a]; n != nil {
		return n
	}
	inner, params := freshOwnTypeParams(g.sub, a.TypeParams())
	rhs := substType(inner, a.Rhs())
	for _, p := range g.paths[a.Obj().Name()] {
		rhs = wrapAtQual(g.sub.pkg, rhs, strings.Split(p.path, "."), p.qual)
	}
	obj := a.Obj()
	n := types2.NewAlias(types2.NewTypeName(obj.Pos(), g.sub.pkg, obj.Name(), nil), rhs)
	n.SetTypeParams(params)
	g.sub.aliases[a] = n
	return n
}

func (g *overlayGraph) applyAliasReadonly(a *types2.Alias) error {
	n, ok := g.sub.aliases[a]
	if !ok {
		return fmt.Errorf("logos: readonly alias %s missing shell", a.Obj().Name())
	}
	paths := g.readonlyPaths[a.Obj().Name()]
	if len(paths) == 0 {
		return nil
	}
	structPaths, genericPaths := splitReadonlyPaths(paths)
	if len(structPaths) != 0 {
		rhs, err := applyReadonlyPaths(g.sub.pkg, g.orig, n.Rhs(), structPaths)
		if err != nil {
			return err
		}
		if rhs != n.Rhs() {
			params := typeParamSlice(n.TypeParams())
			nn := types2.NewAlias(types2.NewTypeName(n.Obj().Pos(), g.sub.pkg, n.Obj().Name(), nil), rhs)
			nn.SetTypeParams(params)
			g.sub.aliases[a] = nn
			n = nn
		}
	}
	for _, p := range genericPaths {
		steps := strings.Split(p, ".")
		head := steps[0]
		params := typeParamSlice(n.TypeParams())
		idx, ok := parsePathIndex(head[1:])
		if !ok || idx < 0 || idx >= len(params) {
			return fmt.Errorf("logos: bad readonly path %q (bad alias type parameter index)", p)
		}
		nt, err := markReadonlyAtPath(g.sub.pkg, g.orig, params[idx].Constraint(), steps[1:])
		if err != nil {
			return err
		}
		if nt != params[idx].Constraint() {
			params[idx].SetConstraint(nt)
		}
	}
	return nil
}

type typeSubst struct {
	pkg     *types2.Package
	tparams map[*types2.TypeParam]*types2.TypeParam
	named   map[*types2.Named]*types2.Named
	aliases map[*types2.Alias]*types2.Alias
	graph   *overlayGraph
}

func typeParamSlice(l *types2.TypeParamList) []*types2.TypeParam {
	if l == nil || l.Len() == 0 {
		return nil
	}
	out := make([]*types2.TypeParam, l.Len())
	for i := range l.Len() {
		out[i] = l.At(i)
	}
	return out
}

func freshTypeParams(sub *typeSubst, olds []*types2.TypeParam) ([]*types2.TypeParam, map[*types2.TypeParam]*types2.TypeParam) {
	news := make([]*types2.TypeParam, len(olds))
	delta := make(map[*types2.TypeParam]*types2.TypeParam, len(olds))
	for i, o := range olds {
		obj := types2.NewTypeName(o.Obj().Pos(), sub.pkg, o.Obj().Name(), nil)
		news[i] = types2.NewTypeParam(obj, nil)
		delta[o] = news[i]
	}
	ext := make(map[*types2.TypeParam]*types2.TypeParam, len(sub.tparams)+len(delta))
	maps.Copy(ext, sub.tparams)
	maps.Copy(ext, delta)
	inner := &typeSubst{pkg: sub.pkg, tparams: ext, named: sub.named, aliases: sub.aliases, graph: sub.graph}
	for i, o := range olds {
		if b := o.Constraint(); b != nil {
			news[i].SetConstraint(substType(inner, b))
		}
	}
	return news, ext
}

func substType(sub *typeSubst, t types2.Type) types2.Type {
	if t == nil {
		return nil
	}
	if p, ok := t.(*types2.TypeParam); ok {
		if n, ok := sub.tparams[p]; ok {
			return n
		}
		return t
	}
	if n, ok := t.(*types2.Named); ok {
		if nn, ok := sub.named[n]; ok {
			return nn
		}
		args := n.TypeArgs()
		if args == nil || args.Len() == 0 {
			return t
		}
		targs := make([]types2.Type, args.Len())
		changed := sub.named[n.Origin()] != nil
		for i := range args.Len() {
			a := substType(sub, args.At(i))
			targs[i] = a
			if a != args.At(i) {
				changed = true
			}
		}
		if !changed {
			return t
		}
		origin := n.Origin()
		if nn, ok := sub.named[origin]; ok {
			origin = nn
		}
		inst, err := types2.Instantiate(nil, origin, targs, false)
		if err != nil {
			panic("overlay named instantiation invariant: " + err.Error())
		}
		return inst
	}
	switch u := t.(type) {
	case *types2.Basic:
		return t
	case *types2.Array:
		e := substType(sub, u.Elem())
		if e == u.Elem() {
			return t
		}
		return types2.NewArray(e, u.Len())
	case *types2.Slice:
		e := substType(sub, u.Elem())
		if e == u.Elem() {
			return t
		}
		return types2.NewSlice(e)
	case *types2.Map:
		k := substType(sub, u.Key())
		e := substType(sub, u.Elem())
		if k == u.Key() && e == u.Elem() {
			return t
		}
		return types2.NewMap(k, e)
	case *types2.Pointer:
		e := substType(sub, u.Elem())
		if e == u.Elem() {
			return t
		}
		return types2.NewPointer(e)
	case *types2.Chan:
		e := substType(sub, u.Elem())
		if e == u.Elem() {
			return t
		}
		return types2.NewChan(u.Dir(), e)
	case *types2.Option:
		e := substType(sub, u.Elem)
		if e == u.Elem {
			return t
		}
		return &types2.Option{Elem: e}
	case *types2.ConstView:
		e := substType(sub, u.Elem)
		if e == u.Elem {
			return t
		}
		return &types2.ConstView{Elem: e, ConstMap: u.ConstMap}
	case *types2.Struct:
		fields := make([]*types2.Var, u.NumFields())
		tags := make([]string, u.NumFields())
		changed := false
		for i := range u.NumFields() {
			f := u.Field(i)
			tags[i] = u.Tag(i)
			e := substType(sub, f.Type())
			if e == f.Type() {
				fields[i] = f
				continue
			}
			nf := types2.NewField(f.Pos(), sub.pkg, f.Name(), e, f.Embedded())
			fields[i] = nf
			changed = true
		}
		if !changed {
			return t
		}
		return types2.NewStruct(fields, tags)
	case *types2.Signature:
		return substSignature(sub, u)
	case *types2.Interface:
		methods := make([]*types2.Func, u.NumExplicitMethods())
		changed := false
		for i := range u.NumExplicitMethods() {
			m := u.ExplicitMethod(i)
			sig, ok := m.Type().(*types2.Signature)
			if !ok {
				methods[i] = m
				continue
			}
			ns := substSignature(sub, sig)
			if ns == sig {
				methods[i] = m
				continue
			}
			nf := types2.NewFunc(m.Pos(), sub.pkg, m.Name(), ns)
			copyFuncReadonly(nf, m)
			methods[i] = nf
			changed = true
		}
		embeddeds := make([]types2.Type, u.NumEmbeddeds())
		for i := range u.NumEmbeddeds() {
			e := substType(sub, u.EmbeddedType(i))
			embeddeds[i] = e
			if e != u.EmbeddedType(i) {
				changed = true
			}
		}
		if !changed {
			return t
		}
		return types2.NewInterfaceType(methods, embeddeds)
	case *types2.Union:
		terms := make([]*types2.Term, u.Len())
		changed := false
		for i := range u.Len() {
			tm := u.Term(i)
			e := substType(sub, tm.Type())
			if e == tm.Type() {
				terms[i] = tm
				continue
			}
			terms[i] = types2.NewTerm(tm.Tilde(), e)
			changed = true
		}
		if !changed || len(terms) == 0 {
			return t
		}
		return types2.NewUnion(terms)
	case *types2.Alias:
		origin := u.Origin()
		mapped := sub.aliases[origin]
		if mapped == nil && sub.graph != nil && sub.graph.affected[origin] {
			mapped = sub.graph.alias(origin)
		}
		if mapped == nil {
			mapped = origin
		}
		args := u.TypeArgs()
		if args.Len() == 0 {
			return mapped
		}
		targs := make([]types2.Type, args.Len())
		changed := mapped != origin
		for i := range args.Len() {
			targs[i] = substType(sub, args.At(i))
			changed = changed || targs[i] != args.At(i)
		}
		if !changed {
			return t
		}
		inst, err := types2.Instantiate(nil, mapped, targs, false)
		if err != nil {
			panic("overlay alias instantiation invariant: " + err.Error())
		}
		return inst
	default:
		return t
	}
}

func substSignature(sub *typeSubst, sig *types2.Signature) *types2.Signature {
	if sub.graph != nil {
		refs := make(map[types2.Type]bool)
		sub.graph.references(sig, refs, make(map[types2.Type]bool))
		changed := false
		for ref := range refs {
			if sub.graph.affected[ref] {
				changed = true
				break
			}
			if p, ok := ref.(*types2.TypeParam); ok && sub.tparams[p] != nil {
				changed = true
				break
			}
		}
		if !changed {
			return sig
		}
	}
	inner := sub
	var newT, newR []*types2.TypeParam
	if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
		var ext map[*types2.TypeParam]*types2.TypeParam
		newT, ext = freshTypeParams(sub, typeParamSlice(tp))
		inner = &typeSubst{pkg: sub.pkg, tparams: ext, named: sub.named, aliases: sub.aliases, graph: sub.graph}
	}
	if rp := sig.RecvTypeParams(); rp != nil && rp.Len() > 0 {
		var ext map[*types2.TypeParam]*types2.TypeParam
		newR, ext = freshTypeParams(inner, typeParamSlice(rp))
		inner = &typeSubst{pkg: sub.pkg, tparams: ext, named: sub.named, aliases: sub.aliases, graph: sub.graph}
	}
	changed := newT != nil || newR != nil
	var nrecv *types2.Var
	if r := sig.Recv(); r != nil {
		if _, lit := r.Type().(*types2.Interface); !lit {
			rt := substType(inner, r.Type())
			if rt != r.Type() {
				changed = true
			}
			nrecv = types2.NewVar(r.Pos(), sub.pkg, r.Name(), rt)
			nrecv.SetKind(r.Kind())
		}
	}
	params, paramsChanged := substTuple(inner, sig.Params())
	results, resultsChanged := substTuple(inner, sig.Results())
	if !changed && !paramsChanged && !resultsChanged {
		return sig
	}
	return types2.NewSignatureType(nrecv, newR, newT, params, results, sig.Variadic())
}

func substTuple(sub *typeSubst, tup *types2.Tuple) (*types2.Tuple, bool) {
	if tup == nil {
		return nil, false
	}
	vars := make([]*types2.Var, tup.Len())
	changed := false
	for i := range tup.Len() {
		v := tup.At(i)
		e := substType(sub, v.Type())
		if e == v.Type() {
			vars[i] = v
			continue
		}
		nv := types2.NewVar(v.Pos(), sub.pkg, v.Name(), e)
		nv.SetKind(v.Kind())
		vars[i] = nv
		changed = true
	}
	if !changed {
		return tup, false
	}
	return types2.NewTuple(vars...), true
}

func cloneInterfaceMethods(pkg *types2.Package, u *types2.Interface, paths []qualPath) *types2.Interface {
	byMethod := map[string][]qualPath{}
	for _, p := range paths {
		steps := strings.Split(p.path, ".")
		if len(steps) < 2 || len(steps[0]) < 2 || steps[0][0] != 'M' {
			continue
		}
		rest := strings.Join(steps[1:], ".")
		name := steps[0][1:]
		byMethod[name] = append(byMethod[name], qualPath{path: rest, qual: p.qual})
	}
	if len(byMethod) == 0 {
		return u
	}
	methods := make([]*types2.Func, u.NumExplicitMethods())
	for i := range u.NumExplicitMethods() {
		m := u.ExplicitMethod(i)
		methods[i] = m
		sig, ok := m.Type().(*types2.Signature)
		if !ok {
			continue
		}
		qp := byMethod[m.Name()]
		if len(qp) == 0 {
			continue
		}
		nsig := sig
		for _, p := range qp {
			nsig = wrapSignaturePos(pkg, nsig, strings.Split(p.path, "."), p.qual).(*types2.Signature)
		}
		nf := types2.NewFunc(m.Pos(), pkg, m.Name(), nsig)
		copyFuncReadonly(nf, m)
		methods[i] = nf
	}
	embeddeds := make([]types2.Type, u.NumEmbeddeds())
	for i := range u.NumEmbeddeds() {
		embeddeds[i] = u.EmbeddedType(i)
	}
	return types2.NewInterfaceType(methods, embeddeds)
}

func hasOwnTypeParams(sig *types2.Signature) bool {
	if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
		return true
	}
	if rp := sig.RecvTypeParams(); rp != nil && rp.Len() > 0 {
		return true
	}
	return false
}

func applyQual(t types2.Type, q qualKind) types2.Type {
	if q == qualConst {
		return toConstView(t)
	}
	return toOption(t)
}

func wrapAtQual(pkg *types2.Package, t types2.Type, steps []string, q qualKind) types2.Type {
	if len(steps) == 0 {
		return applyQual(t, q)
	}
	if t == nil {
		return t
	}
	if o := optAsOption(t); o != nil {
		if len(steps) == 0 {
			return applyQual(t, q)
		}
		if steps[0] != "E" {
			return t
		}
		elem := wrapAtQual(pkg, o.Elem, steps[1:], q)
		if elem == o.Elem {
			return t
		}
		return &types2.Option{Elem: elem}
	}
	if c := constAsView(t); c != nil {
		if len(steps) == 0 {
			return applyQual(t, q)
		}
		if steps[0] != "E" {
			return t
		}
		elem := wrapAtQual(pkg, c.Elem, steps[1:], q)
		if elem == c.Elem {
			return t
		}
		return &types2.ConstView{Elem: elem, ConstMap: c.ConstMap}
	}
	u := types2.Unalias(t)
	switch u := u.(type) {
	case *types2.Slice:
		if steps[0] != "E" {
			return t
		}
		return types2.NewSlice(wrapAtQual(pkg, u.Elem(), steps[1:], q))
	case *types2.Array:
		if steps[0] != "E" {
			return t
		}
		return types2.NewArray(wrapAtQual(pkg, u.Elem(), steps[1:], q), u.Len())
	case *types2.Map:
		if steps[0] == "K" {
			return types2.NewMap(wrapAtQual(pkg, u.Key(), steps[1:], q), u.Elem())
		}
		if steps[0] == "E" {
			return types2.NewMap(u.Key(), wrapAtQual(pkg, u.Elem(), steps[1:], q))
		}
		return t
	case *types2.Pointer:
		if steps[0] != "D" {
			return t
		}
		elem := wrapAtQual(pkg, u.Elem(), steps[1:], q)
		if elem == u.Elem() {
			return t
		}
		return types2.NewPointer(elem)
	case *types2.Chan:
		if steps[0] != "D" {
			return t
		}
		elem := wrapAtQual(pkg, u.Elem(), steps[1:], q)
		if elem == u.Elem() {
			return t
		}
		return types2.NewChan(u.Dir(), elem)
	case *types2.Struct:
		if len(steps[0]) < 2 || steps[0][0] != 'F' {
			return t
		}
		return wrapStructField(pkg, u, steps[0][1:], steps[1:], q)
	case *types2.Signature:
		return wrapSignaturePos(pkg, u, steps, q)
	case *types2.Interface:
		if len(steps[0]) < 2 || steps[0][0] != 'M' {
			return t
		}
		return cloneInterfaceMethods(pkg, u, []qualPath{{path: joinPath(steps), qual: q}})
	case *types2.Named:
		return t
	default:
		return t
	}
}

func wrapStructField(pkg *types2.Package, u *types2.Struct, name string, rest []string, q qualKind) types2.Type {
	idx := -1
	for i := range u.NumFields() {
		if f := u.Field(i); f != nil && f.Name() == name {
			idx = i
			break
		}
	}
	if idx < 0 {
		return u
	}
	fields := make([]*types2.Var, u.NumFields())
	tags := make([]string, u.NumFields())
	changed := false
	for i := range u.NumFields() {
		f := u.Field(i)
		tags[i] = u.Tag(i)
		if i != idx {
			fields[i] = f
			continue
		}
		nt := wrapAtQual(pkg, f.Type(), rest, q)
		if nt == f.Type() {
			fields[i] = f
			continue
		}
		nf := types2.NewField(f.Pos(), pkg, f.Name(), nt, f.Embedded())
		fields[i] = nf
		changed = true
	}
	if !changed {
		return u
	}
	return types2.NewStruct(fields, tags)
}

func wrapSignaturePos(pkg *types2.Package, u *types2.Signature, steps []string, q qualKind) types2.Type {
	if len(steps[0]) < 2 || (steps[0][0] != 'P' && steps[0][0] != 'R') {
		return u
	}
	if hasOwnTypeParams(u) {
		return wrapGenericSignaturePos(pkg, u, steps, q)
	}
	var idx int
	if _, err := fmt.Sscanf(steps[0][1:], "%d", &idx); err != nil {
		return u
	}
	var tup *types2.Tuple
	if steps[0][0] == 'P' {
		tup = u.Params()
	} else {
		tup = u.Results()
	}
	if tup == nil || idx < 0 || idx >= tup.Len() {
		return u
	}
	vars := make([]*types2.Var, tup.Len())
	changed := false
	for i := range tup.Len() {
		v := tup.At(i)
		if i != idx {
			vars[i] = v
			continue
		}
		nt := wrapAtQual(pkg, v.Type(), steps[1:], q)
		if nt == v.Type() {
			vars[i] = v
			continue
		}
		nv := types2.NewVar(v.Pos(), pkg, v.Name(), nt)
		nv.SetKind(v.Kind())
		vars[i] = nv
		changed = true
	}
	if !changed {
		return u
	}
	ntup := types2.NewTuple(vars...)
	var params, results *types2.Tuple
	if steps[0][0] == 'P' {
		params, results = ntup, u.Results()
	} else {
		params, results = u.Params(), ntup
	}
	return types2.NewSignatureType(u.Recv(), nil, nil, params, results, u.Variadic())
}

func wrapGenericSignaturePos(pkg *types2.Package, u *types2.Signature, steps []string, q qualKind) types2.Type {
	var idx int
	if _, err := fmt.Sscanf(steps[0][1:], "%d", &idx); err != nil {
		return u
	}
	base := &typeSubst{pkg: pkg}
	inner, newT := freshOwnTypeParams(base, u.TypeParams())
	inner, newR := freshOwnTypeParams(inner, u.RecvTypeParams())
	full := inner
	var nrecv *types2.Var
	if r := u.Recv(); r != nil {
		rt := substType(full, r.Type())
		nrecv = types2.NewVar(r.Pos(), pkg, r.Name(), rt)
		nrecv.SetKind(r.Kind())
	} else if newR != nil {
		return u
	}
	params, _ := substTuple(full, u.Params())
	results, _ := substTuple(full, u.Results())
	var tup *types2.Tuple
	if steps[0][0] == 'P' {
		tup = params
	} else {
		tup = results
	}
	if tup == nil || idx < 0 || idx >= tup.Len() {
		return u
	}
	v := tup.At(idx)
	nt := wrapAtQual(pkg, v.Type(), steps[1:], q)
	if nt != v.Type() {
		vars := make([]*types2.Var, tup.Len())
		for i := range tup.Len() {
			vars[i] = tup.At(i)
		}
		nv := types2.NewVar(v.Pos(), pkg, v.Name(), nt)
		nv.SetKind(v.Kind())
		vars[idx] = nv
		ntup := types2.NewTuple(vars...)
		if steps[0][0] == 'P' {
			params = ntup
		} else {
			results = ntup
		}
	}
	return types2.NewSignatureType(nrecv, newR, newT, params, results, u.Variadic())
}

func freshOwnTypeParams(sub *typeSubst, l *types2.TypeParamList) (*typeSubst, []*types2.TypeParam) {
	if l == nil || l.Len() == 0 {
		return sub, nil
	}
	news, ext := freshTypeParams(sub, typeParamSlice(l))
	return &typeSubst{pkg: sub.pkg, tparams: ext, named: sub.named, aliases: sub.aliases, graph: sub.graph}, news
}

func toConstView(t types2.Type) types2.Type {
	if t == nil {
		return &types2.ConstView{}
	}
	if c := constAsView(t); c != nil {
		return t
	}
	if _, ok := types2.Unalias(t).(*types2.Map); ok {
		return &types2.ConstView{Elem: t, ConstMap: true}
	}
	return &types2.ConstView{Elem: t}
}
func constMapIndex(recs []facts.ConstMapFact) map[string]facts.ConstMapFact {
	if len(recs) == 0 {
		return nil
	}
	out := make(map[string]facts.ConstMapFact, len(recs))
	for _, r := range recs {
		if r.Object != "" && len(r.Keys) == len(r.Vals) {
			out[r.Object] = r
		}
	}
	return out
}

func (g *overlayGraph) constMapView(name string, t types2.Type) types2.Type {
	rec, ok := g.constMaps[name]
	if !ok || t == nil {
		return t
	}
	if len(rec.Keys) != len(rec.Vals) {
		return t
	}
	if _, ok := types2.Unalias(t).(*types2.Map); !ok {
		return t
	}
	if c := constAsView(t); c != nil {
		return t
	}
	return &types2.ConstView{Elem: t, ConstMap: true}
}

func toOption(t types2.Type) types2.Type {
	if t == nil {
		return &types2.Option{}
	}
	if o, ok := types2.Unalias(t).(*types2.Option); ok {
		return o
	}
	if n, ok := types2.Unalias(t).(*types2.Named); ok {
		if obj := n.Obj(); obj != nil && obj.Name() == "Option" && obj.Pkg() != nil && obj.Pkg().Path() == "logos/rt" {
			if targs := n.TypeArgs(); targs != nil && targs.Len() == 1 {
				return &types2.Option{Elem: targs.At(0)}
			}
		}
	}
	return &types2.Option{Elem: t}
}
