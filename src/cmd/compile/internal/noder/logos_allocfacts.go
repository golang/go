// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/logos/lint"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

type logosCalleeEdge struct {
	key      escape.LogosFuncKey
	name     string
	factName string
	pkgPath  string
	imported bool
}

type logosAllocGraph struct {
	names    map[escape.LogosFuncKey]string
	factName map[escape.LogosFuncKey]string
	allocs   map[escape.LogosFuncKey]bool
	dyn      map[escape.LogosFuncKey]string
	bodyless map[escape.LogosFuncKey]bool
	edges    map[escape.LogosFuncKey][]logosCalleeEdge
	free     map[escape.LogosFuncKey]bool
	decls    []escape.LogosFuncKey
	marked   []lint.ZeroAllocFunc
}

func logosBuildAllocGraph(in *lint.Input, marks []lint.ZeroAllocFunc) *logosAllocGraph {
	g := &logosAllocGraph{
		names:    make(map[escape.LogosFuncKey]string),
		factName: make(map[escape.LogosFuncKey]string),
		allocs:   make(map[escape.LogosFuncKey]bool),
		dyn:      make(map[escape.LogosFuncKey]string),
		bodyless: make(map[escape.LogosFuncKey]bool),
		edges:    make(map[escape.LogosFuncKey][]logosCalleeEdge),
		marked:   marks,
	}
	byObj := make(map[types2.Object]escape.LogosFuncKey)
	for _, f := range in.Files {
		for _, d := range f.DeclList {
			fn, ok := d.(*syntax.FuncDecl)
			if !ok {
				continue
			}
			k := escape.LogosFuncKey{File: f.Pos().RelFilename(), Line: fn.Pos().Line()}
			g.names[k] = lint.DeclName(fn)
			g.factName[k] = lint.DeclName(fn)
			if fn.Body == nil {
				g.bodyless[k] = true
				continue
			}
			if logosGenericRecv(in, fn) {
				g.factName[k] = ""
			}
			g.decls = append(g.decls, k)
			if obj := in.Info.Defs[fn.Name]; obj != nil {
				byObj[obj] = k
			}
		}
	}
	for _, f := range in.Files {
		for _, d := range f.DeclList {
			fn, ok := d.(*syntax.FuncDecl)
			if !ok {
				continue
			}
			k := escape.LogosFuncKey{File: f.Pos().RelFilename(), Line: fn.Pos().Line()}
			if g.bodyless[k] {
				continue
			}
			litVars := logosLiteralVars(in, fn)
			syntax.Inspect(fn, func(n syntax.Node) bool {
				call, ok := n.(*syntax.CallExpr)
				if !ok {
					return true
				}
				g.addCall(in, k, call, byObj, litVars)

				return true
			})
		}
	}
	return g
}

func logosLiteralVars(in *lint.Input, fn *syntax.FuncDecl) map[types2.Object]bool {
	lit := make(map[types2.Object]bool)
	reassigned := make(map[types2.Object]bool)
	mark := func(nm *syntax.Name, byLiteral bool) {
		obj := in.Info.Defs[nm]
		if obj == nil {
			obj = in.Info.Uses[nm]
		}
		if v, isVar := obj.(*types2.Var); isVar && v.Parent() != in.Pkg.Scope() {
			if byLiteral {
				lit[v] = true
			} else {
				reassigned[v] = true
			}
		}
	}
	syntax.Inspect(fn, func(n syntax.Node) bool {
		switch st := n.(type) {
		case *syntax.AssignStmt:
			if st.Rhs == nil {
				return true
			}
			_, byLiteral := stripParens(st.Rhs).(*syntax.FuncLit)
			for _, e := range exprNames(st.Lhs) {
				if nm, ok := e.(*syntax.Name); ok {
					mark(nm, byLiteral)
				}
			}
		case *syntax.VarDecl:
			if st.Values == nil || len(st.NameList) != 1 {
				return true
			}
			_, byLiteral := stripParens(st.Values).(*syntax.FuncLit)
			mark(st.NameList[0], byLiteral)
		}
		return true
	})
	for obj := range reassigned {
		delete(lit, obj)
	}
	return lit
}

func exprNames(e syntax.Expr) []syntax.Expr {
	switch e := e.(type) {
	case nil:
		return nil
	case *syntax.ListExpr:
		return e.ElemList
	case *syntax.Name:
		return []syntax.Expr{e}
	}
	return nil
}

func (g *logosAllocGraph) addCall(in *lint.Input, k escape.LogosFuncKey, call *syntax.CallExpr, byObj map[types2.Object]escape.LogosFuncKey, litVars map[types2.Object]bool) {
	if tv, ok := in.Info.Types[call.Fun]; ok && tv.IsType() {
		return
	}
	if logosTypeExpr(in, stripParens(call.Fun)) {
		return
	}
	switch fun := logosCalleeOperand(in, stripParens(call.Fun)).(type) {
	case *syntax.Name:
		switch obj := in.Info.Uses[fun].(type) {
		case *types2.Func:
			g.addEdge(in, k, obj, byObj)
		case *types2.Var:
			if obj := in.Info.Uses[fun]; !litVars[obj] {
				g.dyn[k] = "dynamic"
			}
		}
	case *syntax.SelectorExpr:
		if sel := in.Info.Selections[fun]; sel != nil {
			if sel.Kind() != types2.MethodVal && sel.Kind() != types2.MethodExpr {
				if sel.Kind() == types2.FieldVal {
					g.dyn[k] = "dynamic"
				}
				return
			}
			if _, isIface := sel.Recv().Underlying().(*types2.Interface); isIface {
				g.dyn[k] = "interface"
				return
			}
			if fn, ok := sel.Obj().(*types2.Func); ok {
				g.addEdge(in, k, fn, byObj)
			}
			return
		}
		if fn, ok := in.Info.Uses[fun.Sel].(*types2.Func); ok {
			g.addEdge(in, k, fn, byObj)
		}
	default:
		g.dyn[k] = "dynamic"
	}
}

func (g *logosAllocGraph) addEdge(in *lint.Input, k escape.LogosFuncKey, fn *types2.Func, byObj map[types2.Object]escape.LogosFuncKey) {
	fn = fn.Origin()
	if fn.Pkg() == nil {
		return
	}
	if fn.Pkg() == in.Pkg {
		key, ok := byObj[fn]
		if !ok || g.bodyless[key] {
			return
		}
		g.edges[k] = append(g.edges[k], logosCalleeEdge{key: key, name: fn.Name()})
		return
	}
	edge := logosCalleeEdge{
		name:     fn.Pkg().Name() + "." + fn.Name(),
		factName: logosTypes2FactName(fn),
		pkgPath:  fn.Pkg().Path(),
		imported: true,
	}
	g.edges[k] = append(g.edges[k], edge)
}

func logosGenericRecv(in *lint.Input, fn *syntax.FuncDecl) bool {
	obj, ok := in.Info.Defs[fn.Name].(*types2.Func)
	if !ok {
		return false
	}
	sig, ok := obj.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	t := sig.Recv().Type()
	if p, isPtr := t.(*types2.Pointer); isPtr {
		t = p.Elem()
	}
	named, isNamed := t.(*types2.Named)
	return isNamed && named.TypeParams() != nil && named.TypeParams().Len() > 0
}

func logosCalleeOperand(in *lint.Input, fun syntax.Expr) syntax.Expr {
	ix, ok := fun.(*syntax.IndexExpr)
	if !ok {
		return fun
	}
	operand := stripParens(ix.X)
	var name *syntax.Name
	switch operand := operand.(type) {
	case *syntax.Name:
		name = operand
	case *syntax.SelectorExpr:
		name = operand.Sel
	}
	if name == nil {
		return fun
	}
	if _, isFunc := in.Info.Uses[name].(*types2.Func); !isFunc {
		return fun
	}
	return operand
}

func logosTypes2FactName(fn *types2.Func) string {
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() == nil {
		return fn.Name()
	}
	t := unconstView(sig.Recv().Type())
	if p, isPtr := types2.Unalias(t).(*types2.Pointer); isPtr {
		t = unconstView(p.Elem())
	}
	if named, isNamed := types2.Unalias(t).(*types2.Named); isNamed && named.TypeParams() != nil && named.TypeParams().Len() > 0 {
		return ""
	}
	rt := unconstView(sig.Recv().Type())
	recv := types2.TypeString(rt, nil)
	if strings.HasPrefix(recv, "*") {
		recv = "(" + recv + ")"
	}
	return recv + "." + fn.Name()
}

func (g *logosAllocGraph) solve() {
	for k := range escape.LogosFuncAllocs {
		g.allocs[k] = true
	}
	free := make(map[escape.LogosFuncKey]bool, len(g.decls))
	for _, k := range g.decls {
		free[k] = true
	}
	for changed := true; changed; {
		changed = false
		for _, k := range g.decls {
			v := g.freeOf(k, free)
			if v != free[k] {
				free[k] = v
				changed = true
			}
		}
	}
	g.free = free
}

func (g *logosAllocGraph) freeOf(k escape.LogosFuncKey, free map[escape.LogosFuncKey]bool) bool {
	if g.allocs[k] || g.dyn[k] != "" || g.bodyless[k] {
		return false
	}
	for _, e := range g.edges[k] {
		if e.imported {
			if !g.factOK(e) {
				return false
			}
			continue
		}
		if !free[e.key] {
			return false
		}
	}
	return true
}

func (g *logosAllocGraph) factOK(e logosCalleeEdge) bool {
	f, ok := facts.Lookup(e.pkgPath)
	if !ok {
		return false
	}
	for _, a := range f.AllocFrees {
		if a.Object == e.factName {
			return true
		}
	}
	return false
}

func (g *logosAllocGraph) findings() []lint.ZeroAllocFinding {
	var out []lint.ZeroAllocFinding
	for _, m := range g.marked {
		k := escape.LogosFuncKey{File: m.File, Line: m.Line}
		if m.Local || g.bodyless[k] {
			continue
		}
		if g.free[k] || g.allocs[k] {
			continue
		}
		out = append(out, lint.ZeroAllocFinding{
			Func:   m.Name,
			File:   m.File,
			Line:   m.Line,
			Col:    1,
			Reason: g.chainReason(k, 2),
		})
	}
	return out
}

func (g *logosAllocGraph) chainReason(k escape.LogosFuncKey, depth int) string {
	if d := g.dyn[k]; d != "" {
		return d + " call may allocate"
	}
	if g.allocs[k] {
		return g.names[k] + " allocates"
	}
	for _, e := range g.edges[k] {
		if e.imported {
			if !g.factOK(e) {
				return e.name + " has no AllocFree fact"
			}
			continue
		}
		if !g.free[e.key] {
			if depth > 0 {
				return g.chainReason(e.key, depth-1)
			}
			return g.names[e.key] + " allocates"
		}
	}
	return g.names[k] + " allocates"
}

func (g *logosAllocGraph) allocFreeObjects() []string {
	var out []string
	for _, k := range g.decls {
		if g.free[k] && g.factName[k] != "" {
			out = append(out, g.factName[k])
		}
	}
	slices.Sort(out)
	return out
}

func logosMergeAllocFacts(objects []string) {
	if len(objects) == 0 {
		return
	}
	f := facts.Current()
	if f == nil {
		f = &facts.Facts{GoPackage: !base.Logos}
		facts.SetCurrent(f)
	}
	for _, o := range objects {
		if slices.ContainsFunc(f.AllocFrees, func(a facts.AllocFreeFact) bool { return a.Object == o }) {
			continue
		}
		f.AllocFrees = append(f.AllocFrees, facts.AllocFreeFact{Object: o})
	}
}

func stripParens(e syntax.Expr) syntax.Expr {
	for {
		p, ok := e.(*syntax.ParenExpr)
		if !ok {
			return e
		}
		e = p.X
	}
}

func logosTypeExpr(in *lint.Input, e syntax.Expr) bool {
	switch e := e.(type) {
	case *syntax.ArrayType, *syntax.SliceType, *syntax.StructType, *syntax.InterfaceType, *syntax.FuncType, *syntax.MapType, *syntax.ChanType:
		return true
	case *syntax.Operation:
		if e.Op != syntax.Mul || e.Y != nil {
			return false
		}
		x := stripParens(e.X)
		if nm, ok := x.(*syntax.Name); ok {
			_, ok := in.Info.Uses[nm].(*types2.TypeName)
			return ok
		}
		if sel, ok := x.(*syntax.SelectorExpr); ok {
			_, ok := in.Info.Uses[sel.Sel].(*types2.TypeName)
			return ok
		}
		tv, ok := in.Info.Types[x]
		return ok && tv.IsType()
	}
	return false
}
