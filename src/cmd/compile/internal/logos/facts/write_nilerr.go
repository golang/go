// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package facts

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func nilErrDriven() bool {
	return base.LogosCfg != "" || base.LogosLintEnabled() || base.LogosAllocFacts
}

func writeNilErrors(pkg *types2.Package, info *types2.Info, files []*syntax.File, f *Facts) {
	if !nilErrDriven() {
		return
	}
	set := AlwaysNilErrorSet(pkg, info, files)
	if len(set) == 0 {
		return
	}
	scope := pkg.Scope()
	if scope == nil {
		return
	}
	for _, name := range scope.Names() {
		switch obj := scope.Lookup(name).(type) {
		case *types2.Func:
			if obj == nil || !obj.Exported() {
				continue
			}
			if set[obj] {
				findOrCreateFnFact(f, name).AlwaysNilError = true
			}
		case *types2.TypeName:
			emitNilErrMethods(f, name, obj, set)
		}
	}
}

func emitNilErrMethods(f *Facts, name string, obj *types2.TypeName, set map[*types2.Func]bool) {
	if obj == nil || obj.IsAlias() {
		return
	}
	named, ok := obj.Type().(*types2.Named)
	if !ok {
		return
	}
	for i := range named.NumMethods() {
		m := named.Method(i)
		if m == nil || !m.Exported() || !set[m] {
			continue
		}
		sig, ok := m.Type().(*types2.Signature)
		if !ok {
			continue
		}
		findOrCreateFnFact(f, MethodObject(name, sig.Recv(), m.Name())).AlwaysNilError = true
	}
}

type nilErrDecl struct {
	fn      *types2.Func
	count   int
	errIdx  []int
	named   map[*types2.Var]bool
	returns []*syntax.ReturnStmt
	assigns []*syntax.AssignStmt
	risky   bool
}

type nilErrScope struct {
	pkg    *types2.Package
	info   *types2.Info
	decls  map[*types2.Func]*nilErrDecl
	proven map[*types2.Func]bool
}

func AlwaysNilErrorSet(pkg *types2.Package, info *types2.Info, files []*syntax.File) map[*types2.Func]bool {
	s := &nilErrScope{pkg: pkg, info: info, decls: make(map[*types2.Func]*nilErrDecl), proven: make(map[*types2.Func]bool)}
	if pkg == nil || info == nil {
		return s.proven
	}
	for _, file := range files {
		for _, decl := range file.DeclList {
			fd, ok := decl.(*syntax.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			s.add(fd)
		}
	}
	for changed := true; changed; {
		changed = false
		for fn, d := range s.decls {
			if s.proven[fn] || !s.holds(d) {
				continue
			}
			s.proven[fn] = true
			changed = true
		}
	}
	return s.proven
}

func (s *nilErrScope) add(fd *syntax.FuncDecl) {
	fn, ok := s.info.Defs[fd.Name].(*types2.Func)
	if !ok {
		return
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || nilErrGeneric(sig) {
		return
	}
	results := sig.Results()
	if results == nil || results.Len() == 0 {
		return
	}
	d := &nilErrDecl{fn: fn, count: results.Len(), named: make(map[*types2.Var]bool)}
	for i := range results.Len() {
		v := results.At(i)
		if !isNilErrType(v.Type()) {
			continue
		}
		d.errIdx = append(d.errIdx, i)
		if v.Name() != "" {
			d.named[v] = true
		}
	}
	if len(d.errIdx) == 0 {
		return
	}
	syntax.Inspect(fd.Body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit:
			return false
		case *syntax.ReturnStmt:
			d.returns = append(d.returns, n)
		case *syntax.AssignStmt:
			d.assigns = append(d.assigns, n)
		}
		return true
	})
	if len(d.named) > 0 {
		d.risky = scanNilErrRisk(s.info, fd.Body, d.named)
	}
	s.decls[fn] = d
}

func scanNilErrRisk(info *types2.Info, body *syntax.BlockStmt, named map[*types2.Var]bool) bool {
	risk := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.CallStmt:
			risk = risk || n.Tok == syntax.Defer
		case *syntax.FuncLit:
			risk = risk || capturesNamed(info, n, named)
		case *syntax.Operation:
			risk = risk || takesNamedAddr(info, n, named)
		case *syntax.RangeClause:
			risk = risk || rangeHitsNamed(info, n, named)
		}
		return true
	})
	return risk
}

func capturesNamed(info *types2.Info, lit *syntax.FuncLit, named map[*types2.Var]bool) bool {
	hit := false
	syntax.Inspect(lit, func(n syntax.Node) bool {
		if nm, ok := n.(*syntax.Name); ok && nilErrTargetsNamed(info, nm, named) {
			hit = true
			return false
		}
		return true
	})
	return hit
}

func takesNamedAddr(info *types2.Info, op *syntax.Operation, named map[*types2.Var]bool) bool {
	if op.Op != syntax.And || op.Y != nil {
		return false
	}
	nm, ok := syntax.Unparen(op.X).(*syntax.Name)
	return ok && nilErrTargetsNamed(info, nm, named)
}

func rangeHitsNamed(info *types2.Info, r *syntax.RangeClause, named map[*types2.Var]bool) bool {
	if r.Def || r.Lhs == nil {
		return false
	}
	return nilErrHitsNamed(info, syntax.UnpackListExpr(r.Lhs), named)
}

func nilErrGeneric(sig *types2.Signature) bool {
	if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
		return true
	}
	recv := sig.Recv()
	if recv == nil {
		return false
	}
	if rt := sig.RecvTypeParams(); rt != nil && rt.Len() > 0 {
		return true
	}
	t := types2.Unalias(recv.Type())
	if p, isPtr := t.(*types2.Pointer); isPtr {
		t = types2.Unalias(p.Elem())
	}
	named, isNamed := t.(*types2.Named)
	if !isNamed {
		return false
	}
	tp := named.TypeParams()
	return tp != nil && tp.Len() > 0
}

func (s *nilErrScope) holds(d *nilErrDecl) bool {
	if d.risky {
		return false
	}
	for _, st := range d.assigns {
		if !s.assignHolds(st, d.named) {
			return false
		}
	}
	for _, ret := range d.returns {
		if !s.returnHolds(ret, d) {
			return false
		}
	}
	return true
}

func (s *nilErrScope) assignHolds(st *syntax.AssignStmt, named map[*types2.Var]bool) bool {
	if len(named) == 0 {
		return true
	}
	lhs := syntax.UnpackListExpr(st.Lhs)
	rhs := syntax.UnpackListExpr(st.Rhs)
	if st.Rhs == nil {
		return !nilErrHitsNamed(s.info, lhs, named)
	}
	if len(rhs) == 1 && len(lhs) > 1 {
		if !nilErrHitsNamed(s.info, lhs, named) {
			return true
		}
		call, ok := syntax.Unparen(rhs[0]).(*syntax.CallExpr)
		return ok && s.provenCall(call)
	}
	if len(lhs) != len(rhs) {
		return !nilErrHitsNamed(s.info, lhs, named)
	}
	for i, e := range lhs {
		nm, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok || !nilErrTargetsNamed(s.info, nm, named) {
			continue
		}
		r := syntax.Unparen(rhs[i])
		if isNilLiteral(r) {
			continue
		}
		call, ok := r.(*syntax.CallExpr)
		if !ok || !s.provenCall(call) {
			return false
		}
	}
	return true
}

func nilErrHitsNamed(info *types2.Info, lhs []syntax.Expr, named map[*types2.Var]bool) bool {
	for _, e := range lhs {
		if nm, ok := syntax.Unparen(e).(*syntax.Name); ok && nilErrTargetsNamed(info, nm, named) {
			return true
		}
	}
	return false
}

func nilErrTargetsNamed(info *types2.Info, nm *syntax.Name, named map[*types2.Var]bool) bool {
	if obj, ok := info.Uses[nm].(*types2.Var); ok && named[obj] {
		return true
	}
	if obj, ok := info.Defs[nm].(*types2.Var); ok && named[obj] {
		return true
	}
	return false
}

func (s *nilErrScope) returnHolds(ret *syntax.ReturnStmt, d *nilErrDecl) bool {
	if ret.Results == nil {
		return true
	}
	elems := syntax.UnpackListExpr(ret.Results)
	if len(elems) == 1 && d.count > 1 {
		call, ok := syntax.Unparen(elems[0]).(*syntax.CallExpr)
		return ok && s.provenCall(call)
	}
	if len(elems) != d.count {
		return false
	}
	for _, i := range d.errIdx {
		e := syntax.Unparen(elems[i])
		if isNilLiteral(e) {
			continue
		}
		call, ok := e.(*syntax.CallExpr)
		if !ok || !s.provenCall(call) {
			return false
		}
	}
	return true
}

func (s *nilErrScope) provenCall(call *syntax.CallExpr) bool {
	fn := resolveNilErrCallee(s.info, call)
	if fn == nil || fn.Pkg() == nil {
		return false
	}
	if fn.Pkg() == s.pkg {
		return s.proven[fn]
	}
	return AlwaysNilCallee(fn)
}

func resolveNilErrCallee(info *types2.Info, call *syntax.CallExpr) *types2.Func {
	switch fun := syntax.Unparen(call.Fun).(type) {
	case *syntax.Name:
		if fn, ok := info.Uses[fun].(*types2.Func); ok {
			return fn
		}
	case *syntax.SelectorExpr:
		if sel := info.Selections[fun]; sel != nil {
			if fn, ok := sel.Obj().(*types2.Func); ok {
				return fn
			}
		}
		if fn, ok := info.Uses[fun.Sel].(*types2.Func); ok {
			return fn
		}
	}
	return nil
}

func AlwaysNilCallee(fn *types2.Func) bool {
	if fn == nil || fn.Pkg() == nil {
		return false
	}
	obj := ObjectName(fn)
	if obj == "" {
		return false
	}
	f, ok := Lookup(fn.Pkg().Path())
	if !ok || f == nil {
		return false
	}
	for _, rec := range f.Fns {
		if rec.Object == obj && rec.AlwaysNilError {
			return true
		}
	}
	return false
}

func isNilLiteral(e syntax.Expr) bool {
	nm, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && nm.Value == "nil"
}

func isNilErrType(t types2.Type) bool {
	named, ok := types2.Unalias(t).(*types2.Named)
	return ok && named.Obj().Pkg() == nil && named.Obj().Name() == "error"
}
