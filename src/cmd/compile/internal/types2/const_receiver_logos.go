// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func LogosProvenKey(fn *Func) (string, bool) {
	return logosProofKey(fn)
}

func LogosReferenceFree(t Type) bool {
	if t == nil {
		return false
	}
	return logosReferenceFree(t)
}

func logosProvenInConf(conf *Config, fn *Func) bool {
	if conf == nil || len(conf.LogosProvenReadonly) == 0 {
		return false
	}
	k, ok := logosProofKey(fn)
	if !ok {
		return false
	}
	return conf.LogosProvenReadonly[k]
}

func LogosStampProvenReadonly(pkg *Package, proven map[string]bool) {
	if pkg == nil || len(proven) == 0 || pkg.Scope() == nil {
		return
	}
	stamp := func(fn *Func) {
		if fn == nil {
			return
		}
		origin := fn.Origin()
		if origin == nil || origin != fn {
			return
		}
		k, ok := logosProofKey(fn)
		if !ok {
			return
		}
		if !proven[k] {
			return
		}
		logosMarkReadonlyCapability(fn)
	}
	for _, name := range pkg.Scope().Names() {
		switch o := pkg.Scope().Lookup(name).(type) {
		case *Func:
			stamp(o.Origin())
		case *TypeName:
			n, ok := o.Type().(*Named)
			if !ok {
				break
			}
			n = n.Origin()
			for i := range n.NumMethods() {
				stamp(n.Method(i).Origin())
			}
		}
	}
}

func logosProofObject(fn *Func) (string, bool) {
	if fn == nil || fn.Origin() != fn {
		return "", false
	}
	sig, ok := fn.Type().(*Signature)
	if !ok {
		return "", false
	}
	if tp := sig.TypeParams(); tp != nil && tp.Len() > 0 {
		return "", false
	}
	if recv := sig.Recv(); recv != nil {
		rt := recv.Type()
		if logosAsConstView(rt) != nil {
			return "", false
		}
		isPtr := false
		if _, ok := Unalias(rt).(*Pointer); ok {
			isPtr = true
		}
		base := rt
		if p, ok := Unalias(base).(*Pointer); ok {
			base = p.Elem()
		}
		n, ok := Unalias(base).(*Named)
		if !ok || n.Obj() == nil {
			return "", false
		}
		if isPtr {
			return "(*" + n.Obj().Name() + ")." + fn.Name(), true
		}
		return n.Obj().Name() + "." + fn.Name(), true
	}
	return fn.Name(), true
}

func LogosSealExportedReadonly(pkg *Package, info *Info, files []*syntax.File) {
	if pkg == nil || pkg.Scope() == nil || info == nil || len(files) == 0 {
		return
	}
	bodies := make(map[*Func]*syntax.FuncDecl)
	for _, file := range files {
		for _, decl := range file.DeclList {
			fd, ok := decl.(*syntax.FuncDecl)
			if !ok || fd == nil || fd.Body == nil {
				continue
			}
			obj, ok := info.Defs[fd.Name].(*Func)
			if !ok || obj == nil || obj.Pkg() != pkg {
				continue
			}
			if obj.Origin() != obj {
				continue
			}
			bodies[obj] = fd
		}
	}
	if len(bodies) == 0 {
		return
	}
	synth := &Checker{
		conf:   &Config{},
		Info:   info,
		pkg:    pkg,
		objMap: make(map[Object]*declInfo, len(bodies)),
	}
	for fn, fd := range bodies {
		synth.objMap[fn] = &declInfo{fdecl: fd}
	}
	for _, name := range pkg.Scope().Names() {
		tn, ok := pkg.Scope().Lookup(name).(*TypeName)
		if !ok || tn == nil {
			continue
		}
		named, ok := tn.Type().(*Named)
		if !ok || named == nil {
			continue
		}
		named = named.Origin()
		if named == nil || named.Obj() == nil || named.Obj().Pkg() != pkg {
			continue
		}
		for i := range named.NumMethods() {
			m := named.Method(i)
			if m == nil {
				continue
			}
			fn := m.Origin()
			if fn == nil || fn != m || !fn.Exported() {
				continue
			}
			if fn.Pkg() != pkg {
				continue
			}
			sig, ok := fn.Type().(*Signature)
			if !ok || sig == nil || sig.Recv() == nil {
				continue
			}
			if logosAsConstView(sig.Recv().Type()) != nil {
				logosMarkReadonlyCapability(fn)
				continue
			}
			if fn.LogosReadonly() {
				continue
			}
			if _, ok := bodies[fn]; !ok {
				continue
			}
			if synth.logosReceiverReadonly(fn) && !fn.LogosReadonly() {
				logosMarkReadonlyCapability(fn)
			}
		}
	}
}

func logosProofKey(fn *Func) (string, bool) {
	if fn == nil || fn.Pkg() == nil {
		return "", false
	}
	obj, ok := logosProofObject(fn)
	if !ok {
		return "", false
	}
	path := fn.Pkg().Path()
	if path == "" {
		return "", false
	}
	return path + "\x00" + obj, true
}

func logosProvenInConfForCheck(check *Checker, fn *Func) bool {
	if check == nil || check.conf == nil {
		return false
	}
	return logosProvenInConf(check.conf, fn)
}

func (check *Checker) logosReceiverReadonly(obj *Func) bool {
	if check == nil || check.conf == nil || check.Info == nil || obj == nil {
		return false
	}
	sig, ok := obj.Type().(*Signature)
	if !ok || sig.Recv() == nil {
		return false
	}
	decl := obj.Origin()
	if decl == nil {
		return false
	}
	if decl.Pkg() != check.pkg {
		return check.logosImportedReadonly(decl)
	}
	rt := sig.Recv().Type()
	if logosAsConstView(rt) != nil {
		return true
	}
	if logosReferenceFree(rt) {
		return true
	}
	if logosProvenInConfForCheck(check, decl) {
		logosMarkReadonlyCapability(decl)
		return true
	}
	di, ok := check.objMap[decl]
	if !ok || di == nil || di.fdecl == nil || di.fdecl.Body == nil {
		return false
	}
	if check.conf.IgnoreFuncBodies {
		return false
	}
	s := &logosRecvScan{
		check:  check,
		memo:   make(map[logosRecvKey]bool),
		active: make(map[logosRecvKey]bool),
	}
	ok = !s.summarize(decl, -1)
	if ok {
		logosMarkReadonlyCapability(decl)
	}
	return ok
}

type logosRecvKey struct {
	fn  *Func
	arg int
}

type logosRecvScan struct {
	check  *Checker
	memo   map[logosRecvKey]bool
	active map[logosRecvKey]bool

	tainted map[Object]bool
	mut     map[*Var]bool
	root    *Var
	results []*Var
	slots   []Type
	changed bool
}

func (s *logosRecvScan) summarize(fn *Func, arg int) bool {
	key := logosRecvKey{fn: fn, arg: arg}
	if done, ok := s.memo[key]; ok {
		return done
	}
	if s.active[key] {
		return true
	}
	sig, ok := fn.Type().(*Signature)
	if !ok {
		return true
	}
	var pv *Var
	if arg < 0 {
		pv = sig.Recv()
	} else if ps := sig.Params(); ps != nil && ps.Len() > 0 {
		j := arg
		if j >= ps.Len() {
			if !sig.Variadic() {
				return true
			}
			j = ps.Len() - 1
		}
		pv = ps.At(j)
	}
	if pv == nil {
		return true
	}
	if logosReferenceFree(pv.Type()) {
		s.memo[key] = false
		return false
	}
	if fn.Pkg() != s.check.pkg {
		return true
	}
	di, ok := s.check.objMap[fn]
	if !ok || di == nil || di.fdecl == nil || di.fdecl.Body == nil {
		return true
	}
	if s.check.conf.IgnoreFuncBodies {
		return true
	}
	savedTainted, savedMut, savedRoot, savedResults, savedSlots, savedChanged :=
		s.tainted, s.mut, s.root, s.results, s.slots, s.changed
	s.tainted = map[Object]bool{pv: true}
	s.mut = map[*Var]bool{}
	s.root = pv
	s.results, s.slots = nil, nil
	if rs := sig.Results(); rs != nil {
		for i := range rs.Len() {
			rv := rs.At(i)
			s.slots = append(s.slots, rv.Type())
			if rv.Name() != "_" && rv.Name() != "" {
				s.results = append(s.results, rv)
			}
		}
	}
	s.active[key] = true
	for i := 0; ; i++ {
		s.changed = false
		s.block(di.fdecl.Body.List)
		if !s.changed {
			break
		}
		if i > 1<<20 {
			s.mut[pv] = true
			break
		}
	}
	res := s.mut[pv]
	delete(s.active, key)
	s.memo[key] = res
	s.tainted, s.mut, s.root, s.results, s.slots, s.changed =
		savedTainted, savedMut, savedRoot, savedResults, savedSlots, savedChanged
	return res
}

func (s *logosRecvScan) resolve(id *syntax.Name) Object {
	if id == nil || s.check.Info == nil {
		return nil
	}
	if obj, ok := s.check.Info.Defs[id]; ok && obj != nil {
		return obj
	}
	return s.check.Info.Uses[id]
}

func (s *logosRecvScan) selectionOf(sel *syntax.SelectorExpr) *Selection {
	if s.check.Info == nil || s.check.Info.Selections == nil {
		return nil
	}
	return s.check.Info.Selections[sel]
}

func (s *logosRecvScan) isPackageVar(obj Object) bool {
	v, ok := obj.(*Var)
	if !ok || v == nil || s.check.pkg == nil {
		return false
	}
	return s.check.pkg.Scope().Lookup(v.Name()) == Object(v)
}

func (s *logosRecvScan) markRoot() {
	if s.root != nil && !s.mut[s.root] {
		s.mut[s.root] = true
		s.changed = true
	}
}

func (s *logosRecvScan) taintVar(obj Object) {
	if obj == nil {
		s.markRoot()
		return
	}
	if !s.tainted[obj] {
		s.tainted[obj] = true
		s.changed = true
	}
}

func (s *logosRecvScan) varLeaks(obj Object) bool {
	v, ok := obj.(*Var)
	if !ok || v == nil || v.Name() == "_" || !s.tainted[Object(v)] {
		return false
	}
	if v.IsField() {
		return true
	}
	t := v.Type()
	if t == nil {
		return true
	}
	return !logosReferenceFree(t)
}

func (s *logosRecvScan) varTypeBearing(v *Var) bool {
	t := v.Type()
	if t == nil {
		return true
	}
	return !logosReferenceFree(t)
}

func (s *logosRecvScan) mentionsTaintedRef(n syntax.Node) bool {
	found := false
	syntax.Inspect(n, func(x syntax.Node) bool {
		if found {
			return false
		}
		if id, ok := x.(*syntax.Name); ok {
			if id.Value == "_" || id.Value == "nil" {
				return true
			}
			obj := s.resolve(id)
			if obj == nil {
				if Universe.Lookup(id.Value) == nil {
					found = true
					return false
				}
				return true
			}
			if s.varLeaks(obj) {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

func (s *logosRecvScan) refBearing(e syntax.Expr) bool {
	t := s.check.Info.TypeOf(e)
	if t == nil {
		return true
	}
	return !logosReferenceFree(t)
}

func (s *logosRecvScan) taintedRef(e syntax.Expr) bool {
	if e == nil {
		return false
	}
	switch n := syntax.Unparen(e).(type) {
	case nil:
		return false
	case *syntax.Name:
		if n.Value == "_" || n.Value == "nil" {
			return false
		}
		obj := s.resolve(n)
		if obj == nil {
			return Universe.Lookup(n.Value) == nil
		}
		if !s.tainted[obj] {
			return false
		}
		return s.refBearing(e)
	case *syntax.SelectorExpr:
		return s.taintedRef(n.X) && s.refBearing(e)
	case *syntax.IndexExpr:
		return s.taintedRef(n.X) && s.refBearing(e)
	case *syntax.SliceExpr:
		return s.taintedRef(n.X) && s.refBearing(e)
	case *syntax.AssertExpr:
		return s.taintedRef(n.X) && s.refBearing(e)
	case *syntax.TypeSwitchGuard:
		return s.taintedRef(n.X) && s.refBearing(e)
	case *syntax.Operation:
		if n.Y == nil {
			if n.Op == syntax.And {
				return s.addrTainted(n.X)
			}
			return s.taintedRef(n.X) && s.refBearing(e)
		}
		return (s.taintedRef(n.X) || s.taintedRef(n.Y)) && s.refBearing(e)
	case *syntax.CallExpr:
		return s.callResultTainted(n)
	case *syntax.CompositeLit:
		for _, el := range n.ElemList {
			if s.taintedRef(el) {
				return true
			}
		}
		return false
	case *syntax.KeyValueExpr:
		return s.taintedRef(n.Key) || s.taintedRef(n.Value)
	case *syntax.FuncLit:
		return s.capturesTainted(n.Body)
	case *syntax.ListExpr:
		for _, el := range n.ElemList {
			if s.taintedRef(el) {
				return true
			}
		}
		return false
	default:
		return s.mentionsTaintedRef(e) && s.refBearing(e)
	}
}

func (s *logosRecvScan) addrTainted(x syntax.Expr) bool {
	switch n := syntax.Unparen(x).(type) {
	case *syntax.CompositeLit:
		return s.taintedRef(x)
	case *syntax.Name:
		if n.Value == "_" || n.Value == "nil" {
			return false
		}
		obj := s.resolve(n)
		if obj == nil {
			return Universe.Lookup(n.Value) == nil
		}
		return s.tainted[obj]
	default:
		if rv := s.rootVarOf(x); rv != nil {
			return s.tainted[rv]
		}
		return s.mentionsTaintedRef(x)
	}
}

func (s *logosRecvScan) rootVarOf(e syntax.Expr) *Var {
	switch n := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if v, ok := s.resolve(n).(*Var); ok && !v.IsField() {
			return v
		}
		return nil
	case *syntax.SelectorExpr:
		return s.rootVarOf(n.X)
	case *syntax.IndexExpr:
		return s.rootVarOf(n.X)
	case *syntax.SliceExpr:
		return s.rootVarOf(n.X)
	case *syntax.Operation:
		if n.Op == syntax.Mul && n.Y == nil {
			return s.rootVarOf(n.X)
		}
		return nil
	default:
		return nil
	}
}

func (s *logosRecvScan) capturesTainted(body *syntax.BlockStmt) bool {
	if body == nil {
		return false
	}
	found := false
	syntax.Inspect(body, func(x syntax.Node) bool {
		if found {
			return false
		}
		if lit, ok := x.(*syntax.FuncLit); ok && lit.Body != nil && lit.Body != body {
			if s.capturesTainted(lit.Body) {
				found = true
				return false
			}
			return false
		}
		if id, ok := x.(*syntax.Name); ok {
			if id.Value != "_" && id.Value != "nil" && s.varLeaks(s.resolve(id)) {
				found = true
				return false
			}
		}
		return true
	})
	return found
}

func (s *logosRecvScan) splitCall(call *syntax.CallExpr) (fun syntax.Expr, recv syntax.Expr, args []syntax.Expr) {
	args = call.ArgList
	fun = syntax.Unparen(call.Fun)
	if idx, ok := fun.(*syntax.IndexExpr); ok {
		if s.resolvesToFunc(idx.X) {
			fun = syntax.Unparen(idx.X)
		} else {
			return idx, nil, args
		}
	}
	if sel, ok := fun.(*syntax.SelectorExpr); ok {
		if choice := s.selectionOf(sel); choice != nil {
			if fn, ok := choice.Obj().(*Func); ok {
				_ = fn
				if choice.Kind() == MethodExpr {
					if len(args) > 0 {
						return sel, args[0], args[1:]
					}
					return sel, nil, args
				}
				return sel, sel.X, args
			}
		}
	}
	return fun, nil, args
}

func (s *logosRecvScan) resolvesToFunc(e syntax.Expr) bool {
	e = syntax.Unparen(e)
	if id, ok := e.(*syntax.Name); ok {
		_, ok := s.resolve(id).(*Func)
		return ok
	}
	if sel, ok := e.(*syntax.SelectorExpr); ok {
		if choice := s.selectionOf(sel); choice != nil {
			_, ok := choice.Obj().(*Func)
			return ok
		}
	}
	return false
}

func (s *logosRecvScan) callResultTainted(call *syntax.CallExpr) bool {
	if !s.refBearing(call) {
		return false
	}
	fun, recv, args := s.splitCall(call)
	if name, ok := fun.(*syntax.Name); ok {
		switch obj := s.resolve(name); obj.(type) {
		case *TypeName:
			if len(args) == 1 {
				return s.taintedRef(args[0])
			}
			return false
		case *Builtin:
			return s.builtinResultTainted(obj.Name(), call)
		case *Func:
		case *Var:
			if s.tainted[obj] {
				return true
			}
		case nil:
			if s.taintedRef(name) {
				return true
			}
		}
	} else if lit, ok := fun.(*syntax.FuncLit); ok {
		if s.capturesTainted(lit.Body) {
			return true
		}
	} else if fun != nil {
		if s.taintedRef(fun) {
			return true
		}
	}
	if recv != nil && s.taintedRef(recv) {
		return true
	}
	for _, a := range args {
		if s.taintedRef(a) {
			return true
		}
	}
	return false
}

func (s *logosRecvScan) builtinResultTainted(name string, call *syntax.CallExpr) bool {
	switch name {
	case "append":
		return s.anyTainted(call.ArgList)
	case "len", "cap", "copy", "make", "new", "print", "println",
		"min", "max", "real", "imag", "complex", "clear", "delete", "close", "panic":
		return false
	}
	return true
}

func (s *logosRecvScan) anyTainted(args []syntax.Expr) bool {
	for _, a := range args {
		if s.taintedRef(a) {
			return true
		}
	}
	return false
}

func (s *logosRecvScan) callEffects(call *syntax.CallExpr) {
	fun, recv, args := s.splitCall(call)
	for _, a := range args {
		s.exprEffects(a)
	}
	if recv != nil {
		s.exprEffects(recv)
	}
	switch f := fun.(type) {
	case *syntax.Name:
		switch o := s.resolve(f).(type) {
		case *Builtin:
			s.builtinEffects(o.Name(), call)
		case *Func:
			s.directCall(nil, o, args)
		case *TypeName:
		case *Var:
			if s.taintedRef(f) || s.anyTainted(args) {
				s.markRoot()
			}
		case nil:
			if s.taintedRef(f) || s.anyTainted(args) {
				s.markRoot()
			}
		default:
			if s.anyTainted(args) {
				s.markRoot()
			}
		}
		return
	case *syntax.SelectorExpr:
		if choice := s.selectionOf(f); choice != nil {
			if fn, ok := choice.Obj().(*Func); ok {
				s.directCall(recv, fn, args)
				return
			}
			s.exprEffects(f.X)
			if s.taintedRef(f.X) || s.anyTainted(args) {
				s.markRoot()
			}
			return
		}
		if id, ok := syntax.Unparen(f.X).(*syntax.Name); ok {
			if _, isPkg := s.resolve(id).(*PkgName); isPkg {
				if s.anyTainted(args) {
					s.markRoot()
				}
				return
			}
		}
		s.exprEffects(f.X)
		if s.taintedRef(f) || s.anyTainted(args) {
			s.markRoot()
		}
		return
	case *syntax.FuncLit:
		s.closureCall(f, args)
		return
	default:
		s.exprEffects(fun)
		if s.taintedRef(fun) || s.anyTainted(args) {
			s.markRoot()
		}
		return
	}
}

func (s *logosRecvScan) directCall(recv syntax.Expr, fn *Func, args []syntax.Expr) {
	fn = fn.Origin()
	sig, ok := fn.Type().(*Signature)
	if !ok {
		if (recv != nil && s.taintedRef(recv)) || s.anyTainted(args) {
			s.markRoot()
		}
		return
	}
	if fn.Pkg() != s.check.pkg {
		if (recv != nil && s.taintedRef(recv)) || s.anyTainted(args) {
			s.markRoot()
		}
		return
	}
	if sig.Recv() != nil && recv != nil && s.taintedRef(recv) {
		if s.summarize(fn, -1) {
			s.markRoot()
		}
	}
	ps := sig.Params()
	if ps == nil || ps.Len() == 0 {
		return
	}
	for i, a := range args {
		j := i
		if j >= ps.Len() {
			if !sig.Variadic() {
				if s.taintedRef(a) {
					s.markRoot()
				}
				continue
			}
			j = ps.Len() - 1
		}
		if s.taintedRef(a) && s.summarize(fn, j) {
			s.markRoot()
		}
	}
}

func (s *logosRecvScan) builtinEffects(name string, call *syntax.CallExpr) {
	args := call.ArgList
	switch name {
	case "append", "copy":
		if len(args) > 0 && s.taintedRef(args[0]) {
			s.markRoot()
		}
	case "delete", "clear", "close", "panic":
		if s.anyTainted(args) {
			s.markRoot()
		}
	}
}

func (s *logosRecvScan) closureCall(lit *syntax.FuncLit, args []syntax.Expr) {
	if lit == nil {
		if s.anyTainted(args) {
			s.markRoot()
		}
		return
	}
	if lit.Type != nil {
		n := len(args)
		for i, fld := range lit.Type.ParamList {
			if fld == nil || fld.Name == nil || i >= n {
				continue
			}
			if s.taintedRef(args[i]) {
				s.bindName(fld.Name, true)
			}
		}
		if n > len(lit.Type.ParamList) && len(lit.Type.ParamList) > 0 {
			if last := lit.Type.ParamList[len(lit.Type.ParamList)-1]; last != nil && last.Name != nil {
				for _, a := range args[len(lit.Type.ParamList):] {
					if s.taintedRef(a) {
						s.bindName(last.Name, true)
						break
					}
				}
			}
		}
	} else if s.anyTainted(args) {
		s.markRoot()
	}
	s.closureEffects(lit)
}

func (s *logosRecvScan) closureEffects(lit *syntax.FuncLit) {
	if lit == nil || lit.Body == nil {
		return
	}
	saved := s.slots
	s.slots = nil
	s.block(lit.Body.List)
	s.slots = saved
}

func (s *logosRecvScan) exprEffects(e syntax.Expr) {
	if e == nil {
		return
	}
	syntax.Inspect(e, func(x syntax.Node) bool {
		switch x := x.(type) {
		case *syntax.CallExpr:
			s.callEffects(x)
			return false
		case *syntax.FuncLit:
			s.closureEffects(x)
			return false
		case *syntax.AssignStmt:
			s.assign(x)
			return false
		case *syntax.DeclStmt:
			s.declList(x.DeclList)
			return false
		case *syntax.SendStmt:
			s.send(x.Chan, x.Value)
			return false
		case *syntax.ReturnStmt:
			s.ret(x.Results)
			return false
		case *syntax.RangeClause:
			s.rangeClause(x)
			return false
		case *syntax.Operation:
			if x.Op == syntax.Recv && x.Y == nil {
				if s.taintedRef(x.X) {
					s.markRoot()
				}
			}
			return true
		}
		return true
	})
}

func (s *logosRecvScan) block(list []syntax.Stmt) {
	for _, st := range list {
		if st != nil {
			s.stmt(st)
		}
	}
}

func (s *logosRecvScan) stmt(st syntax.Stmt) {
	switch st := st.(type) {
	case nil:
		return
	case *syntax.EmptyStmt, *syntax.BranchStmt:
		return
	case *syntax.ExprStmt:
		s.exprEffects(st.X)
	case *syntax.SendStmt:
		s.send(st.Chan, st.Value)
	case *syntax.AssignStmt:
		s.assign(st)
	case *syntax.CallStmt:
		if call, ok := syntax.Unparen(st.Call).(*syntax.CallExpr); ok {
			s.callEffects(call)
			if st.Tok == syntax.Go && s.mentionsTaintedRef(st.Call) {
				s.markRoot()
			}
		} else {
			s.exprEffects(st.Call)
			if s.mentionsTaintedRef(st.Call) {
				s.markRoot()
			}
		}
	case *syntax.ReturnStmt:
		s.ret(st.Results)
	case *syntax.DeclStmt:
		s.declList(st.DeclList)
	case *syntax.IfStmt:
		s.simple(st.Init)
		s.exprEffects(st.Cond)
		if st.Then != nil {
			s.block(st.Then.List)
		}
		if st.Else != nil {
			s.stmt(st.Else)
		}
	case *syntax.ForStmt:
		s.simple(st.Init)
		s.exprEffects(st.Cond)
		s.simple(st.Post)
		if st.Body != nil {
			s.block(st.Body.List)
		}
	case *syntax.RangeClause:
		s.rangeClause(st)
	case *syntax.SwitchStmt:
		s.simple(st.Init)
		if g, ok := syntax.Unparen(st.Tag).(*syntax.TypeSwitchGuard); ok {
			s.typeSwitch(st, g)
		} else {
			s.exprEffects(st.Tag)
			for _, c := range st.Body {
				if c == nil {
					continue
				}
				s.exprEffects(c.Cases)
				s.block(c.Body)
			}
		}
	case *syntax.SelectStmt:
		for _, c := range st.Body {
			if c == nil {
				continue
			}
			s.simple(c.Comm)
			s.block(c.Body)
		}
	case *syntax.LabeledStmt:
		s.stmt(st.Stmt)
	case *syntax.BlockStmt:
		s.block(st.List)
	case *syntax.YieldStmt:
		s.exprEffects(st.Value)
		if s.taintedRef(st.Value) {
			s.markRoot()
		}
	case *syntax.ScopeDeferStmt:
		s.exprEffects(st.Call)
	case *syntax.GroupStmt:
		s.exprEffects(st.Ctx)
		s.exprEffects(st.Limit)
		if st.Body != nil {
			if s.mentionsTaintedRef(st.Body) {
				s.markRoot()
			}
			s.block(st.Body.List)
		}
	case *syntax.GroupChild:
		if st.Body != nil {
			if s.mentionsTaintedRef(st.Body) {
				s.markRoot()
			}
			s.block(st.Body.List)
		}
	default:
		if s.mentionsTaintedRef(st) {
			s.markRoot()
		}
	}
}

func (s *logosRecvScan) simple(st syntax.SimpleStmt) {
	switch st := st.(type) {
	case nil:
		return
	case *syntax.EmptyStmt:
		return
	case *syntax.AssignStmt:
		s.assign(st)
	case *syntax.SendStmt:
		s.send(st.Chan, st.Value)
	case *syntax.ExprStmt:
		s.exprEffects(st.X)
	case *syntax.RangeClause:
		s.rangeClause(st)
	default:
		if s.mentionsTaintedRef(st) {
			s.markRoot()
		}
	}
}

func (s *logosRecvScan) send(ch, val syntax.Expr) {
	s.exprEffects(ch)
	s.exprEffects(val)
	if s.taintedRef(ch) || s.taintedRef(val) {
		s.markRoot()
	}
}

func (s *logosRecvScan) ret(results syntax.Expr) {
	if results == nil {
		for _, rv := range s.results {
			if rv != nil && s.tainted[Object(rv)] {
				s.markRoot()
			}
		}
		return
	}
	s.exprEffects(results)
	exprs := syntax.UnpackListExpr(results)
	for i, e := range exprs {
		if s.taintedRef(e) && !s.aliasFreeReturn(e, i, len(exprs)) {
			s.markRoot()
		}
	}
}

func (s *logosRecvScan) aliasFreeReturn(e syntax.Expr, i, n int) bool {
	var result Type
	if n == len(s.slots) {
		result = s.slots[i]
	}
	return LogosAliasFreeReturn(s.check.Info.TypeOf(e), result)
}

func (s *logosRecvScan) declList(list []syntax.Decl) {
	for _, d := range list {
		vd, ok := d.(*syntax.VarDecl)
		if !ok || vd == nil || vd.Values == nil {
			continue
		}
		s.exprEffects(vd.Values)
		vals := syntax.UnpackListExpr(vd.Values)
		if len(vals) == 1 && len(vd.NameList) != 1 {
			t := s.taintedRef(vals[0])
			for _, name := range vd.NameList {
				s.bindName(name, t)
			}
			continue
		}
		for i, name := range vd.NameList {
			t := false
			if i < len(vals) {
				t = s.taintedRef(vals[i])
			} else if len(vals) > 0 {
				t = s.taintedRef(vals[len(vals)-1])
			}
			s.bindName(name, t)
		}
	}
}

func (s *logosRecvScan) bindName(name *syntax.Name, t bool) {
	if name == nil || name.Value == "_" || !t {
		return
	}
	obj, ok := s.resolve(name).(*Var)
	if !ok || obj == nil || obj.IsField() {
		s.markRoot()
		return
	}
	if s.isPackageVar(obj) {
		s.markRoot()
		return
	}
	s.taintVar(obj)
}

func (s *logosRecvScan) assign(st *syntax.AssignStmt) {
	s.exprEffects(st.Lhs)
	if st.Rhs == nil {
		for _, e := range syntax.UnpackListExpr(st.Lhs) {
			s.store(e, false)
		}
		return
	}
	s.exprEffects(st.Rhs)
	if st.Op != 0 && st.Op != syntax.Def {
		for _, e := range syntax.UnpackListExpr(st.Lhs) {
			s.store(e, false)
		}
		return
	}
	lhs := syntax.UnpackListExpr(st.Lhs)
	rhs := syntax.UnpackListExpr(st.Rhs)
	if len(rhs) == 1 && len(lhs) != 1 {
		t := s.taintedRef(rhs[0])
		for _, e := range lhs {
			s.store(e, t)
		}
		return
	}
	for i, e := range lhs {
		t := false
		if i < len(rhs) {
			t = s.taintedRef(rhs[i])
		} else if len(rhs) > 0 {
			t = s.taintedRef(rhs[len(rhs)-1])
		}
		s.store(e, t)
	}
}

func (s *logosRecvScan) store(lhs syntax.Expr, rhsTainted bool) {
	root, base, ok := s.storeTarget(lhs)
	if !ok {
		if rhsTainted || s.mentionsTaintedRef(lhs) {
			s.markRoot()
		}
		return
	}
	if root == nil {
		if rhsTainted || (base != nil && s.taintedRef(base)) || s.mentionsTaintedRef(lhs) {
			s.markRoot()
		}
		return
	}
	if root.Name() == "_" {
		return
	}
	if root.IsField() {
		if rhsTainted || s.mentionsTaintedRef(lhs) {
			s.markRoot()
		}
		return
	}
	if s.isPackageVar(root) {
		if rhsTainted || (base != nil && s.taintedRef(base)) {
			s.markRoot()
		}
		return
	}
	if base != nil && s.taintedRef(base) {
		s.markRoot()
	}
	if rhsTainted {
		s.taintVar(root)
	}
}

func (s *logosRecvScan) storeTarget(lhs syntax.Expr) (root *Var, base syntax.Expr, ok bool) {
	switch n := syntax.Unparen(lhs).(type) {
	case *syntax.Name:
		if v, isVar := s.resolve(n).(*Var); isVar && v != nil {
			return v, nil, true
		}
		return nil, nil, false
	case *syntax.SelectorExpr:
		if _, isPtr := s.check.Info.TypeOf(n.X).(*Pointer); isPtr {
			return s.rootVarOf(n.X), n.X, true
		}
		r, b, ok := s.storeTarget(n.X)
		if !ok {
			return nil, nil, false
		}
		return r, b, true
	case *syntax.IndexExpr:
		if t := s.check.Info.TypeOf(n.X); t != nil {
			if _, isArr := t.Underlying().(*Array); isArr {
				return s.storeTarget(n.X)
			}
		}
		return s.rootVarOf(n.X), n.X, true
	case *syntax.Operation:
		if n.Op == syntax.Mul && n.Y == nil {
			return s.rootVarOf(n.X), n.X, true
		}
		return nil, nil, false
	default:
		return nil, nil, false
	}
}

func (s *logosRecvScan) rangeClause(rc *syntax.RangeClause) {
	if rc == nil {
		return
	}
	s.exprEffects(rc.X)
	xTainted := s.taintedRef(rc.X)
	if rc.Lhs == nil {
		if xTainted {
			s.markRoot()
		}
		return
	}
	if xTainted {
		xt := s.check.Info.TypeOf(rc.X)
		if xt == nil {
			s.markRoot()
		} else {
			switch xt.Underlying().(type) {
			case *Chan, *Signature:
				s.markRoot()
			}
		}
	}
	for _, e := range syntax.UnpackListExpr(rc.Lhs) {
		name, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok || name == nil || name.Value == "_" {
			continue
		}
		obj, isVar := s.resolve(name).(*Var)
		if !isVar || obj == nil || obj.IsField() {
			if xTainted {
				s.markRoot()
			}
			continue
		}
		if s.isPackageVar(obj) {
			if xTainted {
				s.markRoot()
			}
			continue
		}
		if xTainted && s.varTypeBearing(obj) {
			s.taintVar(obj)
		}
	}
}

func (s *logosRecvScan) typeSwitch(st *syntax.SwitchStmt, g *syntax.TypeSwitchGuard) {
	s.exprEffects(g.X)
	xTainted := s.taintedRef(g.X)
	for _, c := range st.Body {
		if c == nil {
			continue
		}
		s.exprEffects(c.Cases)
		if g.Lhs != nil && xTainted {
			obj, _ := s.check.Info.Implicits[c].(*Var)
			if obj == nil || obj.IsField() || s.isPackageVar(obj) {
				s.markRoot()
			} else if s.varTypeBearing(obj) {
				s.taintVar(obj)
			}
		}
		s.block(c.Body)
	}
}

func logosIfaceMethodReadonly(fn *Func) bool {
	if fn == nil {
		return false
	}
	if fn.LogosReadonly() {
		return true
	}
	if origin := fn.Origin(); origin != nil && origin != fn && origin.LogosReadonly() {
		return true
	}
	return false
}

func logosMarkReadonlyCapability(fn *Func) {
	if fn == nil {
		return
	}
	fn.SetLogosReadonly(true)
}

func logosConcreteReadonlySatisfiedNil(f, m *Func) bool {
	if !logosIfaceMethodReadonly(m) {
		return true
	}
	if f == nil {
		return false
	}
	if f.LogosReadonly() {
		return true
	}
	return false
}

func (check *Checker) logosImportedReadonly(fn *Func) bool {
	if fn == nil {
		return false
	}
	if fn.LogosReadonly() {
		return true
	}
	if origin := fn.Origin(); origin != nil && origin != fn && origin.LogosReadonly() {
		return true
	}
	return false
}

type logosCallReadonlyDeferred struct {
	check    *Checker
	fn       *Func
	pos      syntax.Pos
	baseExpr syntax.Expr
	name     string
}

func (d *logosCallReadonlyDeferred) cleanup() {
	if d == nil || d.check == nil || d.fn == nil {
		return
	}
	if !d.check.logosReceiverReadonly(d.fn) {
		d.check.errorf(d.pos, IncompatibleAssign, "%s cannot call %s through const view %s", logosConstMutation, d.name, ExprString(d.baseExpr))
	}
}

func (check *Checker) logosDeferReadonlyCheck(fn *Func, pos syntax.Pos, baseExpr syntax.Expr, name string) {
	if check == nil || fn == nil {
		return
	}
	check.needsCleanup(&logosCallReadonlyDeferred{
		check:    check,
		fn:       fn,
		pos:      pos,
		baseExpr: baseExpr,
		name:     name,
	})
}

func (check *Checker) logosReadonlySelect(x *operand, e *syntax.SelectorExpr, name string) bool {
	if !check.conf.Logos || x.mode() == typexpr || !x.isValid() {
		return false
	}
	container := x.typ()
	base, descendants, readonly := logosReadonlyBase(container)
	indirect := false
	if p, ok := base.Underlying().(*Pointer); ok {
		if _, d, r := logosReadonlyBase(p.base); d || r {
			container = p.base
			base, descendants, readonly = logosReadonlyBase(container)
			indirect = true
		}
	}
	if !descendants || readonly {
		return false
	}
	obj, index, throughPointer := lookupFieldOrMethod(base, x.mode() == variable || indirect, check.pkg, name, false)
	switch obj := obj.(type) {
	case *Var:
		projectedContainer := container
		current := base
		throughReference := false
		for _, field := range index {
			if p, ok := current.Underlying().(*Pointer); ok {
				current = p.base
				throughReference = true
			}
			s, ok := current.Underlying().(*Struct)
			if !ok {
				return false
			}
			current = s.fields[field].Type()
		}
		check.recordSelection(e, FieldVal, base, obj, index, indirect || throughPointer)
		if x.mode() == variable || indirect || throughPointer {
			x.mode_ = variable
		} else {
			x.mode_ = value
		}
		x.typ_ = logosReadonlyProject(projectedContainer, obj.Type(), throughReference)
		x.expr = e
		return true
	case *Func:
		check.objDecl(obj)
		sig, _ := obj.Type().(*Signature)
		if sig == nil || sig.Recv() == nil {
			return false
		}
		if !logosReferenceFree(sig.Recv().Type()) && !check.logosReceiverReadonly(obj) {
			check.errorf(e.Sel, IncompatibleAssign, "%s cannot call %s through readonly descendants", logosConstEscape, name)
			x.invalidate()
			return true
		}
		x.typ_ = base
		if indirect {
			x.typ_ = &Pointer{base: base}
		}
		return false
	}
	return false
}
