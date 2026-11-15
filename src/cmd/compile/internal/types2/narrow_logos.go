// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"maps"
	"slices"
)

const logosResultBeforeErr = "[correctness::result_before_error_check]"

func (check *Checker) logosEnterFunc(body *syntax.BlockStmt) {
	check.logosEnterFuncBody(body)
	if !check.conf.Logos {
		return
	}
	check.logosLitSave = append(check.logosLitSave, nil)
}

func (check *Checker) logosEnterLit(body *syntax.BlockStmt) {
	check.logosEnterFuncBody(body)
	if !check.conf.Logos {
		return
	}
	check.logosStripTainted()
}

func (check *Checker) logosEnterFuncBody(body *syntax.BlockStmt) {
	if !check.conf.Logos {
		return
	}
	outermost := len(check.logosDisqual) == 0
	dq := &logosDisqualWalk{
		stage:     make(map[*syntax.LambdaExpr]bool),
		addr:      make(map[string]bool),
		inLoop:    make(map[string]bool),
		inClos:    make(map[string]bool),
		refInClos: make(map[string]bool),
		assignOut: make(map[string]bool),
	}
	if sig := check.sig; sig != nil {
		for _, vars := range []*Tuple{sig.params, sig.results} {
			for i := range vars.Len() {
				dq.params = append(dq.params, vars.At(i).name)
			}
		}
		if sig.recv != nil {
			dq.params = append(dq.params, sig.recv.name)
		}
	}
	if body != nil {
		syntax.Walk(body, dq)
	}
	set := make(map[string]bool, len(dq.addr)+len(dq.inLoop)+len(dq.inClos))
	for name := range dq.addr {
		set[name] = true
	}
	for name := range dq.inLoop {
		set[name] = true
	}
	for name := range dq.inClos {
		set[name] = true
	}
	check.logosDisqual = append(check.logosDisqual, set)
	if body != nil {
		if check.Info.LogosDisqual == nil {
			check.Info.LogosDisqual = make(map[*syntax.BlockStmt][]map[string]bool)
		}
		check.Info.LogosDisqual[body] = slices.Clone(check.logosDisqual)
	}
	if outermost {
		check.logosEFBody = body
		tainted := make(map[string]bool)
		for name := range dq.refInClos {
			if dq.assignOut[name] {
				tainted[name] = true
			}
		}
		check.logosTainted = tainted
	}
}

func (check *Checker) logosStripTainted() {
	var saved map[*Var]int
	for name := range check.logosTainted {
		obj := check.lookup(name)
		v, ok := obj.(*Var)
		if !ok || v.pkg != check.pkg {
			continue
		}
		if check.logosNarrow[v] > 0 {
			if saved == nil {
				saved = make(map[*Var]int)
			}
			saved[v] = check.logosNarrow[v]
			delete(check.logosNarrow, v)
		}
	}
	check.logosLitSave = append(check.logosLitSave, saved)
}

func (check *Checker) logosRestoreTainted() {
	n := len(check.logosLitSave)
	if n == 0 {
		return
	}
	saved := check.logosLitSave[n-1]
	check.logosLitSave = check.logosLitSave[:n-1]
	for v, c := range saved {
		check.logosNarrow[v] = c
	}
}

func (check *Checker) logosExitFunc() {
	if !check.conf.Logos {
		return
	}
	check.logosRestoreTainted()
	if n := len(check.logosDisqual); n > 0 {
		if n == 1 {
			check.logosErrorFirst(check.logosEFBody)
		}
		check.logosDisqual = check.logosDisqual[:n-1]
	}
}

type logosDisqualWalk struct {
	stack     []syntax.Node
	scopes    []map[string]bool
	params    []string
	stage     map[*syntax.LambdaExpr]bool
	loop      int
	clos      int
	lam       int
	addr      map[string]bool
	inLoop    map[string]bool
	inClos    map[string]bool
	refInClos map[string]bool
	assignOut map[string]bool
}

func (w *logosDisqualWalk) Visit(node syntax.Node) syntax.Visitor {
	if node == nil {
		w.leave()
		return w
	}
	w.declare(node)
	switch t := node.(type) {
	case *syntax.FuncLit, *syntax.GroupChild:
		w.clos++
	case *syntax.CallExpr:
		w.markStage(t)
	case *syntax.LambdaExpr:
		if w.stage[t] {
			w.lam++
		}
	case *syntax.ForStmt:
		w.loop++
	case *syntax.Operation:
		if t.Op == syntax.And && t.Y == nil {
			if id, ok := syntax.Unparen(t.X).(*syntax.Name); ok && id.Value != "_" {
				w.addr[id.Value] = true
			}
		}
	case *syntax.Name:
		if w.clos > 0 && t.Value != "_" {
			w.refInClos[t.Value] = true
		}
	case *syntax.AssignStmt:
		w.assign(t)
	case *syntax.RangeClause:
		if !t.Def {
			for _, name := range logosAssignNames(t.Lhs) {
				w.inLoop[name] = true
				w.assigned(name)
			}
		}
	}
	w.stack = append(w.stack, node)
	return w
}

func (w *logosDisqualWalk) leave() {
	top := w.stack[len(w.stack)-1]
	w.stack = w.stack[:len(w.stack)-1]
	switch top := top.(type) {
	case *syntax.FuncLit, *syntax.GroupChild:
		w.clos--
	case *syntax.LambdaExpr:
		if w.stage[top] {
			w.lam--
		}
	case *syntax.ForStmt:
		w.loop--
	case *syntax.BlockStmt, *syntax.CaseClause, *syntax.CommClause:
		w.scopes = w.scopes[:len(w.scopes)-1]
	}
}

func (w *logosDisqualWalk) declare(node syntax.Node) {
	switch t := node.(type) {
	case *syntax.FuncLit:
		for _, f := range slices.Concat(t.Type.ParamList, t.Type.ResultList) {
			if f.Name != nil {
				w.params = append(w.params, f.Name.Value)
			}
		}
	case *syntax.LambdaExpr:
		if _, ok := t.Body.(*syntax.BlockStmt); ok {
			for _, name := range t.Params {
				w.params = append(w.params, name.Value)
			}
		}
	case *syntax.BlockStmt, *syntax.CaseClause, *syntax.CommClause:
		scope := make(map[string]bool, len(w.params))
		for _, name := range w.params {
			scope[name] = true
		}
		w.params = nil
		w.scopes = append(w.scopes, scope)
	case *syntax.VarDecl:
		for _, name := range t.NameList {
			w.scopes[len(w.scopes)-1][name.Value] = true
		}
	}
}

func (w *logosDisqualWalk) assign(s *syntax.AssignStmt) {
	header := false
	switch w.stack[len(w.stack)-1].(type) {
	case *syntax.IfStmt, *syntax.ForStmt, *syntax.SwitchStmt:
		header = true
	}
	scope := w.scopes[len(w.scopes)-1]
	for _, name := range logosAssignNames(s.Lhs) {
		if s.Op == syntax.Def {
			redeclared := scope[name] && !header
			scope[name] = scope[name] || !header
			if !redeclared {
				continue
			}
		}
		if w.loop > 0 {
			w.inLoop[name] = true
		}
		w.assigned(name)
	}
}

func (w *logosDisqualWalk) assigned(name string) {
	if w.clos > 0 || w.lam > 0 {
		w.inClos[name] = true
	} else {
		w.assignOut[name] = true
	}
}

func (w *logosDisqualWalk) markStage(call *syntax.CallExpr) {
	if logosStageCalls(call) == nil {
		return
	}
	for _, arg := range call.ArgList {
		if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
			w.stage[lam] = true
		}
	}
}

func logosAssignNames(lhs syntax.Expr) []string {
	var names []string
	for _, e := range syntax.UnpackListExpr(lhs) {
		if id, ok := syntax.Unparen(e).(*syntax.Name); ok && id.Value != "_" {
			names = append(names, id.Value)
		}
	}
	return names
}

func (check *Checker) logosDisqualified(name string) bool {
	for _, set := range check.logosDisqual {
		if set[name] {
			return true
		}
	}
	return false
}

func (check *Checker) logosNarrowable(v *Var) bool {
	if v == nil || v.pkg != check.pkg || v.typ == nil {
		return false
	}
	switch v.Kind() {
	case LocalVar, RecvVar, ParamVar, ResultVar:
	default:
		return false
	}
	if logosAsOption(v.typ) == nil {
		return false
	}
	return !check.logosDisqualified(v.name)
}

func (check *Checker) logosNilTestOf(cond syntax.Expr) (v *Var, isNeq bool, ok bool) {
	op, ok := syntax.Unparen(cond).(*syntax.Operation)
	if !ok || (op.Op != syntax.Eql && op.Op != syntax.Neq) || op.Y == nil {
		return nil, false, false
	}
	var name *syntax.Name
	if check.isNil(op.X) {
		name, _ = syntax.Unparen(op.Y).(*syntax.Name)
	} else if check.isNil(op.Y) {
		name, _ = syntax.Unparen(op.X).(*syntax.Name)
	} else {
		return nil, false, false
	}
	if name == nil || name.Value == "_" {
		return nil, false, false
	}
	obj := check.lookup(name.Value)
	w, _ := obj.(*Var)
	if w == nil {
		return nil, false, false
	}
	return w, op.Op == syntax.Neq, true
}

func (check *Checker) logosCollectNeNil(cond syntax.Expr) []*Var {
	op, ok := syntax.Unparen(cond).(*syntax.Operation)
	if !ok || op.Y == nil {
		return nil
	}
	if op.Op == syntax.AndAnd {
		left := check.logosCollectNeNil(op.X)
		right := check.logosCollectNeNil(op.Y)
		return append(left, right...)
	}
	if op.Op != syntax.Neq {
		return nil
	}
	v, isNeq, ok := check.logosNilTestOf(cond)
	if !ok || !isNeq {
		return nil
	}
	return []*Var{v}
}

func (check *Checker) logosPushNarrow(vars []*Var) bool {
	if !check.conf.Logos {
		return false
	}
	var frame []*Var
	for _, v := range vars {
		if check.logosNarrowable(v) {
			frame = append(frame, v)
		}
	}
	if len(frame) == 0 {
		return false
	}
	if check.logosNarrow == nil {
		check.logosNarrow = make(map[*Var]int)
	}
	for _, v := range frame {
		check.logosNarrow[v]++
	}
	check.logosNarrowStack = append(check.logosNarrowStack, frame)
	return true
}

func (check *Checker) logosPopNarrow() {
	frames := check.logosNarrowStack
	frame := frames[len(frames)-1]
	check.logosNarrowStack = frames[:len(frames)-1]
	for _, v := range frame {
		if c := check.logosNarrow[v]; c <= 1 {
			delete(check.logosNarrow, v)
		} else {
			check.logosNarrow[v] = c - 1
		}
	}
}

func (check *Checker) logosNarrowBase() int {
	return len(check.logosNarrowStack)
}

func (check *Checker) logosPopToNarrow(base int) {
	for len(check.logosNarrowStack) > base {
		check.logosPopNarrow()
	}
}

func (check *Checker) logosPushIfThen(cond syntax.Expr) bool {
	return check.logosPushNarrow(check.logosCollectNeNil(cond))
}

func (check *Checker) logosPushAndCond(op syntax.Operator, lhs syntax.Expr) bool {
	if !check.conf.Logos || op != syntax.AndAnd {
		return false
	}
	return check.logosPushNarrow(check.logosCollectNeNil(lhs))
}

func (check *Checker) logosAfterStmt(s syntax.Stmt) {
	if !check.conf.Logos {
		return
	}
	ifStmt, ok := s.(*syntax.IfStmt)
	if !ok || ifStmt.Then == nil {
		return
	}
	v, isNeq, ok := check.logosNilTestOf(ifStmt.Cond)
	if !ok {
		return
	}
	if !check.logosExits(ifStmt.Then) {
		return
	}
	if ifStmt.Else != nil && !check.logosExits(ifStmt.Else) && (isNeq || check.logosElseAssigns(ifStmt.Else, v)) {
		return
	}
	if !isNeq {
		check.logosPushNarrow([]*Var{v})
	}
}

func (check *Checker) logosExits(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.ReturnStmt:
		return true
	case *syntax.BranchStmt:
		return s.Tok != syntax.Fallthrough
	case *syntax.BlockStmt:
		list := s.List
		for len(list) > 0 {
			if _, ok := list[len(list)-1].(*syntax.EmptyStmt); !ok {
				break
			}
			list = list[:len(list)-1]
		}
		if len(list) == 0 {
			return false
		}
		return check.logosExits(list[len(list)-1])
	case *syntax.IfStmt:
		return s.Else != nil && check.logosExits(s.Then) && check.logosExits(s.Else)
	case *syntax.LabeledStmt:
		return check.logosExits(s.Stmt)
	case *syntax.ExprStmt:
		if call, ok := syntax.Unparen(s.X).(*syntax.CallExpr); ok {
			if check.isPanic[call] {
				return true
			}
			if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok && name.Value == "panic" {
				return true
			}
		}
		return false
	case *syntax.ForStmt:
		if s.Cond != nil {
			return false
		}
		if _, ok := s.Init.(*syntax.RangeClause); ok {
			return false
		}
		hasBreak := false
		syntax.Inspect(s.Body, func(n syntax.Node) bool {
			if b, ok := n.(*syntax.BranchStmt); ok && b.Tok == syntax.Break {
				hasBreak = true
				return false
			}
			return !hasBreak
		})
		return !hasBreak
	default:
		return false
	}
}

func (check *Checker) logosElseAssigns(els syntax.Stmt, v *Var) bool {
	assigns := false
	syntax.Inspect(els, func(n syntax.Node) bool {
		if assigns {
			return false
		}
		switch t := n.(type) {
		case *syntax.AssignStmt:
			if t.Op != syntax.Def {
				for _, name := range logosAssignNames(t.Lhs) {
					if name == v.name {
						assigns = true
						return false
					}
				}
			}
		case *syntax.RangeClause:
			if !t.Def {
				for _, name := range logosAssignNames(t.Lhs) {
					if name == v.name {
						assigns = true
						return false
					}
				}
			}
		}
		return true
	})
	return assigns
}

func (check *Checker) logosPushIfElse(cond syntax.Expr) bool {
	v, isNeq, ok := check.logosNilTestOf(cond)
	if !ok || isNeq {
		return false
	}
	return check.logosPushNarrow([]*Var{v})
}

func (check *Checker) logosNarrowIdent(x *operand, obj Object) {
	if !check.conf.Logos {
		return
	}
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg {
		return
	}
	if n := len(check.logosLHS); n > 0 && check.logosLHS[n-1] == v {
		return
	}
	if check.logosNarrow[v] > 0 {
		if opt := logosAsOption(v.typ); opt != nil {
			x.typ_ = opt.Elem
		}
	}
}

func (check *Checker) logosStmtEnter(s syntax.Stmt) syntax.Stmt {
	prev := check.logosCurStmt
	check.logosCurStmt = s
	return prev
}

func (check *Checker) logosStmtExit(prev syntax.Stmt) {
	check.logosCurStmt = prev
}

func (check *Checker) logosLHSPush(lhs syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	var v *Var
	if id, ok := syntax.Unparen(lhs).(*syntax.Name); ok && id.Value != "_" {
		v, _ = check.lookup(id.Value).(*Var)
	}
	check.logosLHS = append(check.logosLHS, v)
}

func (check *Checker) logosLHSPop() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosLHS); n > 0 {
		check.logosLHS = check.logosLHS[:n-1]
	}
}

func (check *Checker) logosKillNarrow(lhs syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	id, ok := syntax.Unparen(lhs).(*syntax.Name)
	if !ok || id.Value == "_" {
		return
	}
	obj := check.lookup(id.Value)
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg {
		return
	}
	check.logosKillVar(v)
}

func (check *Checker) logosShortDecl(lhs []syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	for _, e := range lhs {
		id, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok || id.Value == "_" {
			continue
		}
		if alt, _ := check.scope.Lookup(id.Value).(*Var); alt != nil {
			check.logosKillVar(alt)
		}
	}
}

func (check *Checker) logosKillVar(v *Var) {
	if v == nil || v.pkg != check.pkg {
		return
	}
	delete(check.logosNarrow, v)
}

func (check *Checker) logosUnnarrowOperand(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	id, ok := syntax.Unparen(x.expr).(*syntax.Name)
	if !ok {
		return
	}
	obj := check.lookup(id.Value)
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg || check.logosNarrow[v] == 0 {
		return
	}
	if logosAsOption(v.typ) == nil {
		return
	}
	x.typ_ = v.typ
	check.updateExprType(x.expr, v.typ, false)
}

type logosEFEntry struct {
	err    *Var
	name   string
	proven bool
}

type logosEFState struct {
	m map[*Var]logosEFEntry
}

func (s *logosEFState) set(v *Var, e logosEFEntry) *logosEFState {
	m := maps.Clone(s.m)
	if m == nil {
		m = make(map[*Var]logosEFEntry, 1)
	}
	m[v] = e
	return &logosEFState{m}
}

func (s *logosEFState) drop(v *Var) *logosEFState {
	if _, ok := s.m[v]; !ok {
		return s
	}
	m := maps.Clone(s.m)
	delete(m, v)
	return &logosEFState{m}
}

func (s *logosEFState) remap(f func(logosEFEntry) logosEFEntry) *logosEFState {
	var out map[*Var]logosEFEntry
	for v, e := range s.m {
		ne := f(e)
		if ne == e {
			continue
		}
		if out == nil {
			out = maps.Clone(s.m)
		}
		out[v] = ne
	}
	if out == nil {
		return s
	}
	return &logosEFState{out}
}

func (s *logosEFState) prove(errV *Var) *logosEFState {
	return s.remap(func(e logosEFEntry) logosEFEntry {
		e.proven = e.proven || e.err == errV
		return e
	})
}

func (s *logosEFState) detach(errV *Var) *logosEFState {
	return s.remap(func(e logosEFEntry) logosEFEntry {
		if e.err == errV {
			e.err = nil
		}
		return e
	})
}

func (p *logosEFPass) capture(st *logosEFState) *logosEFState {
	if p.assigns == nil {
		p.collectAssigns()
	}
	var out map[*Var]logosEFEntry
	for v, e := range st.m {
		if !e.proven || p.initialized[v] && len(p.assigns[v]) == 1 && !p.addr[v] {
			continue
		}
		if out == nil {
			out = maps.Clone(st.m)
		}
		e.proven = false
		out[v] = e
	}
	if out == nil {
		return st
	}
	return &logosEFState{out}
}

func logosEFJoin(a, b *logosEFState) *logosEFState {
	switch {
	case a == nil:
		return b
	case b == nil || a == b:
		return a
	}
	m := maps.Clone(a.m)
	if m == nil {
		m = make(map[*Var]logosEFEntry, len(b.m))
	}
	for v, eb := range b.m {
		ea, ok := m[v]
		if !ok {
			m[v] = eb
			continue
		}
		if ea.err != eb.err {
			ea.err = nil
		}
		ea.proven = ea.proven && eb.proven
		m[v] = ea
	}
	return &logosEFState{m}
}

func logosEFSame(a, b *logosEFState) bool {
	if a == nil || b == nil {
		return a == b
	}
	return maps.Equal(a.m, b.m)
}

type logosEFFrame struct {
	label    string
	loop     bool
	brk, cnt *logosEFState
}

type logosEFCtx struct {
	labels  map[string]*logosEFState
	frames  []*logosEFFrame
	exit    *logosEFState
	fall    *logosEFState
	changed bool
}

type logosEFEffects struct {
	inside   map[*Var]bool
	assigned map[*Var]bool
	created  map[*Var]string
}

type logosEFPass struct {
	check       *Checker
	body        *syntax.BlockStmt
	seen        map[syntax.Pos]bool
	ctx         *logosEFCtx
	eff         *logosEFEffects
	assigns     map[*Var][]syntax.Expr
	addr        map[*Var]bool
	initialized map[*Var]bool
	tracked     map[*Var]string
}

func (check *Checker) logosErrorFirst(body *syntax.BlockStmt) {
	if !check.conf.Logos || body == nil || check.Uses == nil || check.Defs == nil {
		return
	}
	p := &logosEFPass{check: check, body: body, seen: make(map[syntax.Pos]bool)}
	if p.hasTuple() {
		p.run(body, &logosEFState{})
	}
}

func (p *logosEFPass) hasTuple() bool {
	found := false
	syntax.Inspect(p.body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.AssignStmt:
			if n.Rhs != nil {
				errV, _ := p.tuple(syntax.UnpackListExpr(n.Lhs), n.Rhs)
				found = found || errV != nil
			}
		case *syntax.VarDecl:
			if n.Values != nil {
				errV, _ := p.tuple(logosEFNames(n), n.Values)
				found = found || errV != nil
			}
		}
		return !found
	})
	return found
}

func logosEFNames(d *syntax.VarDecl) []syntax.Expr {
	lhs := make([]syntax.Expr, len(d.NameList))
	for i, name := range d.NameList {
		lhs[i] = name
	}
	return lhs
}

func (p *logosEFPass) run(body *syntax.BlockStmt, in *logosEFState) *logosEFState {
	saved := p.ctx
	ctx := &logosEFCtx{labels: make(map[string]*logosEFState)}
	p.ctx = ctx
	var end *logosEFState
	for {
		ctx.frames, ctx.exit, ctx.fall, ctx.changed = nil, nil, nil, false
		end = p.stmt(body, in)
		if !ctx.changed {
			break
		}
	}
	p.ctx = saved
	return logosEFJoin(end, ctx.exit)
}

func (p *logosEFPass) varOf(e syntax.Expr) *Var {
	id, ok := syntax.Unparen(e).(*syntax.Name)
	if !ok || id.Value == "_" {
		return nil
	}
	obj := p.check.Defs[id]
	if obj == nil {
		obj = p.check.Uses[id]
	}
	v, _ := obj.(*Var)
	return v
}

func (p *logosEFPass) addressRoot(e syntax.Expr) *Var {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.SelectorExpr:
		if sel := p.check.Selections[e]; sel != nil && sel.Kind() == FieldVal && !sel.Indirect() {
			return p.addressRoot(e.X)
		}
	case *syntax.IndexExpr:
		if typ, _, ok := p.check.logosChecked(e.X); ok {
			for _, u := range typeset(typ) {
				if _, ok := u.(*Array); ok {
					return p.addressRoot(e.X)
				}
			}
		}
	default:
		return p.varOf(e)
	}
	return nil
}

func (p *logosEFPass) errVar(e syntax.Expr) *Var {
	v := p.varOf(e)
	if v == nil || v.typ == nil || !Identical(v.typ, universeError) {
		return nil
	}
	return v
}

func (p *logosEFPass) isNil(e syntax.Expr) bool {
	id, ok := e.(*syntax.Name)
	if !ok {
		return false
	}
	_, isNil := p.check.Uses[id].(*Nil)
	return isNil
}

func (p *logosEFPass) nilTest(op *syntax.Operation) (errV *Var, isNeq bool) {
	if op.Y == nil || (op.Op != syntax.Eql && op.Op != syntax.Neq) {
		return nil, false
	}
	x, y := syntax.Unparen(op.X), syntax.Unparen(op.Y)
	switch {
	case p.isNil(y):
		errV = p.errVar(x)
	case p.isNil(x):
		errV = p.errVar(y)
	}
	return errV, op.Op == syntax.Neq
}

func (p *logosEFPass) tuple(lhs []syntax.Expr, rhs syntax.Expr) (errV *Var, vars []*Var) {
	call, ok := syntax.Unparen(rhs).(*syntax.CallExpr)
	if !ok || len(lhs) < 2 {
		return nil, nil
	}
	typ, _, ok := p.check.logosChecked(call)
	tup, isTuple := typ.(*Tuple)
	if !ok || !isTuple || tup.Len() != len(lhs) || !Identical(tup.At(tup.Len()-1).typ, universeError) {
		return nil, nil
	}
	if !p.calleeInLgo(call.Fun) {
		return nil, nil
	}
	errV = p.errVar(lhs[len(lhs)-1])
	if errV == nil {
		return nil, nil
	}
	for _, e := range lhs[:len(lhs)-1] {
		if v := p.varOf(e); v != nil && v != errV {
			vars = append(vars, v)
		}
	}
	return errV, vars
}

func (p *logosEFPass) staticFunc(e syntax.Expr) *Func {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		fn, _ := p.check.Uses[e].(*Func)
		return fn
	case *syntax.SelectorExpr:
		if sel := p.check.Selections[e]; sel != nil {
			fn, _ := sel.Obj().(*Func)
			return fn
		}
		fn, _ := p.check.Uses[e.Sel].(*Func)
		return fn
	case *syntax.IndexExpr:
		return p.staticFunc(e.X)
	}
	return nil
}

func (p *logosEFPass) calleeInLgo(fun syntax.Expr) bool {
	if fn := p.staticFunc(fun); fn != nil {
		return p.check.logosFuncInLgo(fn)
	}
	id, ok := syntax.Unparen(fun).(*syntax.Name)
	if !ok {
		return true
	}
	v, _ := p.check.Uses[id].(*Var)
	if origin := p.sole(v); origin != nil {
		if fn := p.staticFunc(origin); fn != nil {
			return p.check.logosFuncInLgo(fn)
		}
	}
	return true
}

func (p *logosEFPass) sole(v *Var) syntax.Expr {
	if v == nil {
		return nil
	}
	if p.assigns == nil {
		p.collectAssigns()
	}
	if exprs := p.assigns[v]; len(exprs) == 1 && !p.addr[v] {
		return exprs[0]
	}
	return nil
}

func (p *logosEFPass) collectAssigns() {
	p.assigns = make(map[*Var][]syntax.Expr)
	p.addr = make(map[*Var]bool)
	p.initialized = make(map[*Var]bool)
	p.tracked = make(map[*Var]string)
	mark := func(lhs []syntax.Expr, rhs syntax.Expr) {
		errV, vars := p.tuple(lhs, rhs)
		if errV == nil {
			return
		}
		for _, v := range vars {
			if _, ok := p.tracked[v]; !ok {
				p.tracked[v] = errV.name
			}
		}
	}
	note := func(lhs []syntax.Expr, rhs []syntax.Expr, plain bool) {
		for i, l := range lhs {
			var r syntax.Expr
			if plain && len(lhs) == len(rhs) {
				r = rhs[i]
			}
			if v := p.varOf(l); v != nil {
				p.assigns[v] = append(p.assigns[v], r)
				if id, ok := syntax.Unparen(l).(*syntax.Name); ok && p.check.Defs[id] == v {
					p.initialized[v] = true
				}
			}
		}
	}
	syntax.Inspect(p.body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.AssignStmt:
			lhs := syntax.UnpackListExpr(n.Lhs)
			rhs := syntax.UnpackListExpr(n.Rhs)
			note(lhs, rhs, n.Op == 0 || n.Op == syntax.Def)
			if (n.Op == 0 || n.Op == syntax.Def) && len(rhs) == 1 {
				mark(lhs, rhs[0])
			}
		case *syntax.VarDecl:
			if n.Values != nil {
				names := logosEFNames(n)
				note(names, syntax.UnpackListExpr(n.Values), true)
				mark(names, n.Values)
			}
		case *syntax.RangeClause:
			note(syntax.UnpackListExpr(n.Lhs), nil, false)
		case *syntax.Operation:
			if n.Op == syntax.And && n.Y == nil {
				if v := p.addressRoot(n.X); v != nil {
					p.addr[v] = true
				}
			}
		case *syntax.SelectorExpr:
			sel := p.check.Selections[n]
			if sel == nil || sel.Kind() != MethodVal || sel.Indirect() {
				break
			}
			if sel.Obj().(*Func).hasPtrRecv() {
				if v := p.addressRoot(n.X); v != nil {
					p.addr[v] = true
				}
			}
		}
		return true
	})
}

func (p *logosEFPass) read(n *syntax.Name, st *logosEFState) {
	v, _ := p.check.Uses[n].(*Var)
	if v == nil {
		return
	}
	e, ok := st.m[v]
	if !ok || e.proven || p.seen[n.Pos()] {
		return
	}
	p.seen[n.Pos()] = true
	p.check.errorf(n, IncompatibleAssign, "%s read of %s before %s is checked (read %s only after %s is checked)", logosResultBeforeErr, v.name, e.name, v.name, e.name)
}

func (p *logosEFPass) noteAssigned(v *Var) {
	if p.eff != nil && !p.eff.inside[v] {
		p.eff.assigned[v] = true
	}
}

func (p *logosEFPass) noteCreated(v *Var, errName string) {
	if p.eff != nil && !p.eff.inside[v] {
		p.eff.created[v] = errName
	}
}

func (p *logosEFPass) assigned(st *logosEFState, v *Var) *logosEFState {
	p.noteAssigned(v)
	return st.drop(v).detach(v)
}

func (p *logosEFPass) track(lhs []syntax.Expr, rhs syntax.Expr, st *logosEFState) *logosEFState {
	errV, vars := p.tuple(lhs, rhs)
	if errV == nil {
		return st
	}
	for _, v := range vars {
		st = st.set(v, logosEFEntry{err: errV, name: errV.name})
		p.noteCreated(v, errV.name)
	}
	return st
}

func (p *logosEFPass) expr(n syntax.Node, st *logosEFState) *logosEFState {
	if n == nil || st == nil {
		return st
	}
	syntax.Inspect(n, func(x syntax.Node) bool {
		switch x := x.(type) {
		case *syntax.Name:
			p.read(x, st)
		case *syntax.Operation:
			if x.Y == nil || (x.Op != syntax.AndAnd && x.Op != syntax.OrOr) {
				return true
			}
			t, f := p.cond(x, st)
			st = logosEFJoin(t, f)
			return false
		case *syntax.FuncLit:
			st = p.later(x, x.Body, st)
			return false
		case *syntax.LambdaExpr:
			st = p.lambda(x, st)
			return false
		}
		return true
	})
	return st
}

func (p *logosEFPass) cond(e syntax.Expr, st *logosEFState) (t, f *logosEFState) {
	if st == nil {
		return nil, nil
	}
	e = syntax.Unparen(e)
	op, ok := e.(*syntax.Operation)
	if !ok {
		st = p.expr(e, st)
		return st, st
	}
	switch {
	case op.Op == syntax.Not && op.Y == nil:
		t, f = p.cond(op.X, st)
		return f, t
	case op.Op == syntax.AndAnd && op.Y != nil:
		xt, xf := p.cond(op.X, st)
		yt, yf := p.cond(op.Y, xt)
		return yt, logosEFJoin(xf, yf)
	case op.Op == syntax.OrOr && op.Y != nil:
		xt, xf := p.cond(op.X, st)
		yt, yf := p.cond(op.Y, xf)
		return logosEFJoin(xt, yt), yf
	}
	st = p.expr(op, st)
	errV, isNeq := p.nilTest(op)
	if errV == nil {
		return st, st
	}
	if isNeq {
		return st, st.prove(errV)
	}
	return st.prove(errV), st
}

func (p *logosEFPass) stmts(list []syntax.Stmt, st *logosEFState) *logosEFState {
	for _, s := range list {
		st = p.stmt(s, st)
	}
	return st
}

func (p *logosEFPass) stmt(s syntax.Stmt, st *logosEFState) *logosEFState {
	if s == nil {
		return st
	}
	if ls, ok := s.(*syntax.LabeledStmt); ok {
		return p.labeled(ls, st)
	}
	if st == nil {
		return nil
	}
	switch s := s.(type) {
	case *syntax.BlockStmt:
		return p.stmts(s.List, st)
	case *syntax.ExprStmt:
		return p.exprStmt(s, st)
	case *syntax.SendStmt:
		return p.expr(s.Value, p.expr(s.Chan, st))
	case *syntax.DeclStmt:
		return p.decls(s, st)
	case *syntax.AssignStmt:
		return p.assign(s, st)
	case *syntax.BranchStmt:
		return p.branch(s, st)
	case *syntax.ReturnStmt:
		return p.ret(s, st)
	case *syntax.IfStmt:
		return p.ifStmt(s, st)
	case *syntax.ForStmt:
		return p.forStmt(s, "", st)
	case *syntax.SwitchStmt:
		return p.switchStmt(s, "", st)
	case *syntax.SelectStmt:
		return p.selectStmt(s, "", st)
	}
	return p.logosStmt(s, st)
}

func (p *logosEFPass) logosStmt(s syntax.Stmt, st *logosEFState) *logosEFState {
	switch s := s.(type) {
	case *syntax.GroupStmt:
		return p.stmt(s.Body, p.expr(s.Limit, p.expr(s.Ctx, st)))
	case *syntax.GroupChild:
		return p.later(s.Body, s.Body, st)
	case *syntax.ConstIfStmt:
		st = p.expr(s.Cond, st)
		return logosEFJoin(p.stmt(s.Then, st), p.stmt(s.Else, st))
	case *syntax.ConstSwitchStmt:
		out := st
		for _, c := range s.Cases {
			out = logosEFJoin(out, p.stmts(c.Body, st))
		}
		return out
	case *syntax.ConstForStmt:
		return p.loop(s.Body, nil, nil, "", p.expr(s.Range, st), nil)
	}
	return p.expr(s, st)
}

func (p *logosEFPass) labeled(s *syntax.LabeledStmt, st *logosEFState) *logosEFState {
	st = logosEFJoin(st, p.ctx.labels[s.Label.Value])
	if st == nil {
		return nil
	}
	switch inner := s.Stmt.(type) {
	case *syntax.ForStmt:
		return p.forStmt(inner, s.Label.Value, st)
	case *syntax.SwitchStmt:
		return p.switchStmt(inner, s.Label.Value, st)
	case *syntax.SelectStmt:
		return p.selectStmt(inner, s.Label.Value, st)
	}
	return p.stmt(s.Stmt, st)
}

func (p *logosEFPass) exprStmt(s *syntax.ExprStmt, st *logosEFState) *logosEFState {
	st = p.expr(s.X, st)
	call, ok := syntax.Unparen(s.X).(*syntax.CallExpr)
	if !ok {
		return st
	}
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return st
	}
	if b, ok := p.check.Uses[name].(*Builtin); ok && b.Name() == "panic" {
		return nil
	}
	return st
}

func (p *logosEFPass) decls(s *syntax.DeclStmt, st *logosEFState) *logosEFState {
	for _, d := range s.DeclList {
		vd, ok := d.(*syntax.VarDecl)
		if !ok {
			continue
		}
		st = p.expr(vd.Values, st)
		names := logosEFNames(vd)
		for _, name := range names {
			if v := p.varOf(name); v != nil {
				st = p.assigned(st, v)
			}
		}
		if vd.Values != nil {
			st = p.track(names, vd.Values, st)
		}
	}
	return st
}

func (p *logosEFPass) assign(s *syntax.AssignStmt, st *logosEFState) *logosEFState {
	lhs := syntax.UnpackListExpr(s.Lhs)
	rhs := syntax.UnpackListExpr(s.Rhs)
	plain := s.Op == 0 || s.Op == syntax.Def
	for i, r := range rhs {
		if !(s.Op == 0 && len(lhs) == len(rhs) && logosEFBlank(lhs[i]) && logosEFIsName(r)) {
			st = p.expr(r, st)
		}
	}
	for _, l := range lhs {
		if !(plain && s.Rhs != nil && logosEFIsName(l)) {
			st = p.expr(l, st)
		}
	}
	for _, l := range lhs {
		if v := p.varOf(l); v != nil {
			st = p.assigned(st, v)
		}
	}
	if plain && len(rhs) == 1 {
		st = p.track(lhs, rhs[0], st)
	}
	return st
}

func logosEFIsName(e syntax.Expr) bool {
	_, ok := syntax.Unparen(e).(*syntax.Name)
	return ok
}

func logosEFBlank(e syntax.Expr) bool {
	id, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && id.Value == "_"
}

func (p *logosEFPass) ret(s *syntax.ReturnStmt, st *logosEFState) *logosEFState {
	results := syntax.UnpackListExpr(s.Results)
	passed := make(map[*Var]bool, len(results))
	for _, r := range results {
		if v := p.varOf(r); v != nil {
			passed[v] = true
		}
	}
	for _, r := range results {
		if e, ok := st.m[p.varOf(r)]; ok && e.err != nil && passed[e.err] {
			continue
		}
		st = p.expr(r, st)
	}
	p.ctx.exit = logosEFJoin(p.ctx.exit, st)
	return nil
}

func (p *logosEFPass) branch(s *syntax.BranchStmt, st *logosEFState) *logosEFState {
	switch s.Tok {
	case syntax.Break:
		if f := p.frame(s.Label, false); f != nil {
			f.brk = logosEFJoin(f.brk, st)
		}
	case syntax.Continue:
		if f := p.frame(s.Label, true); f != nil {
			f.cnt = logosEFJoin(f.cnt, st)
		}
	case syntax.Goto:
		old := p.ctx.labels[s.Label.Value]
		if joined := logosEFJoin(old, st); !logosEFSame(old, joined) {
			p.ctx.labels[s.Label.Value] = joined
			p.ctx.changed = true
		}
	case syntax.Fallthrough:
		p.ctx.fall = logosEFJoin(p.ctx.fall, st)
	}
	return nil
}

func (p *logosEFPass) pushFrame(label string, loop bool) *logosEFFrame {
	f := &logosEFFrame{label: label, loop: loop}
	p.ctx.frames = append(p.ctx.frames, f)
	return f
}

func (p *logosEFPass) popFrame() {
	p.ctx.frames = p.ctx.frames[:len(p.ctx.frames)-1]
}

func (p *logosEFPass) frame(label *syntax.Name, loop bool) *logosEFFrame {
	for i := len(p.ctx.frames) - 1; i >= 0; i-- {
		f := p.ctx.frames[i]
		switch {
		case label != nil && f.label == label.Value:
			return f
		case label == nil && (f.loop || !loop):
			return f
		}
	}
	return nil
}

func (p *logosEFPass) ifStmt(s *syntax.IfStmt, st *logosEFState) *logosEFState {
	st = p.stmt(s.Init, st)
	t, f := p.cond(s.Cond, st)
	return logosEFJoin(p.stmt(s.Then, t), p.stmt(s.Else, f))
}

func (p *logosEFPass) forStmt(s *syntax.ForStmt, label string, st *logosEFState) *logosEFState {
	rc, isRange := s.Init.(*syntax.RangeClause)
	if !isRange {
		return p.loop(s.Body, s.Cond, s.Post, label, p.stmt(s.Init, st), nil)
	}
	return p.loop(s.Body, nil, nil, label, p.expr(rc.X, st), rc)
}

func (p *logosEFPass) loop(body *syntax.BlockStmt, cond syntax.Expr, post syntax.SimpleStmt, label string, st *logosEFState, rc *syntax.RangeClause) *logosEFState {
	if st == nil {
		return nil
	}
	head := st
	for {
		f := p.pushFrame(label, true)
		in, out := head, (*logosEFState)(nil)
		switch {
		case rc != nil:
			in, out = p.rangeAssign(rc, head), head
		case cond != nil:
			in, out = p.cond(cond, head)
		}
		end := p.stmt(post, logosEFJoin(p.stmt(body, in), f.cnt))
		p.popFrame()
		next := logosEFJoin(head, end)
		if logosEFSame(next, head) {
			return logosEFJoin(out, f.brk)
		}
		head = next
	}
}

func (p *logosEFPass) rangeAssign(rc *syntax.RangeClause, st *logosEFState) *logosEFState {
	if rc.Lhs == nil || rc.Def {
		return st
	}
	for _, l := range syntax.UnpackListExpr(rc.Lhs) {
		if v := p.varOf(l); v != nil {
			st = p.assigned(st, v)
		} else {
			st = p.expr(l, st)
		}
	}
	return st
}

func (p *logosEFPass) switchStmt(s *syntax.SwitchStmt, label string, st *logosEFState) *logosEFState {
	st = p.stmt(s.Init, st)
	var tagErr *Var
	switch tag := s.Tag.(type) {
	case *syntax.TypeSwitchGuard:
		tagErr = p.errVar(tag.X)
		st = p.expr(tag.X, st)
	case nil:
	default:
		tagErr = p.errVar(tag)
		st = p.expr(tag, st)
	}
	f := p.pushFrame(label, false)
	enter := make([]*logosEFState, len(s.Body))
	miss := st
	hasDefault := false
	for i, c := range s.Body {
		if c.Cases == nil {
			hasDefault = true
			continue
		}
		for _, e := range syntax.UnpackListExpr(c.Cases) {
			var hit *logosEFState
			hit, miss = p.caseCond(s.Tag, tagErr, e, miss)
			enter[i] = logosEFJoin(enter[i], hit)
		}
	}
	var out, fall *logosEFState
	for i, c := range s.Body {
		if c.Cases == nil {
			enter[i] = miss
		}
		p.ctx.fall = nil
		out = logosEFJoin(out, p.stmts(c.Body, logosEFJoin(enter[i], fall)))
		fall = p.ctx.fall
	}
	p.ctx.fall = nil
	p.popFrame()
	if !hasDefault {
		out = logosEFJoin(out, miss)
	}
	return logosEFJoin(out, f.brk)
}

func (p *logosEFPass) caseCond(tag syntax.Expr, tagErr *Var, e syntax.Expr, miss *logosEFState) (hit, rest *logosEFState) {
	switch {
	case tag == nil:
		return p.cond(e, miss)
	case tagErr != nil && p.isNil(syntax.Unparen(e)):
		return miss.prove(tagErr), miss
	}
	miss = p.expr(e, miss)
	return miss, miss
}

func (p *logosEFPass) selectStmt(s *syntax.SelectStmt, label string, st *logosEFState) *logosEFState {
	f := p.pushFrame(label, false)
	var out *logosEFState
	for _, c := range s.Body {
		out = logosEFJoin(out, p.stmts(c.Body, p.stmt(c.Comm, st)))
	}
	p.popFrame()
	return logosEFJoin(out, f.brk)
}

func (p *logosEFPass) defsIn(n syntax.Node) map[*Var]bool {
	inside := make(map[*Var]bool)
	syntax.Inspect(n, func(x syntax.Node) bool {
		if id, ok := x.(*syntax.Name); ok {
			if v, _ := p.check.Defs[id].(*Var); v != nil {
				inside[v] = true
			}
		}
		return true
	})
	return inside
}

func (p *logosEFPass) lateCaptures(st *logosEFState, lit syntax.Node, inside map[*Var]bool) *logosEFState {
	if p.assigns == nil {
		p.collectAssigns()
	}
	syntax.Inspect(lit, func(n syntax.Node) bool {
		id, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		v, _ := p.check.Uses[id].(*Var)
		if v == nil || inside[v] {
			return true
		}
		if _, ok := st.m[v]; ok {
			return true
		}
		if errName, ok := p.tracked[v]; ok {
			st = st.set(v, logosEFEntry{name: errName})
		}
		return true
	})
	return st
}

func (p *logosEFPass) later(lit syntax.Node, body *syntax.BlockStmt, st *logosEFState) *logosEFState {
	if body == nil {
		return st
	}
	inside := p.defsIn(lit)
	saved := p.eff
	eff := &logosEFEffects{inside: inside, assigned: make(map[*Var]bool), created: make(map[*Var]string)}
	p.eff = eff
	p.run(body, p.lateCaptures(p.capture(st), lit, inside))
	p.eff = saved
	for v := range eff.assigned {
		st = st.detach(v)
		p.noteAssigned(v)
	}
	for v, errName := range eff.created {
		st = st.set(v, logosEFEntry{name: errName})
		p.noteCreated(v, errName)
	}
	return st
}

func (p *logosEFPass) lambda(lam *syntax.LambdaExpr, st *logosEFState) *logosEFState {
	body, isBlock := lam.Body.(*syntax.BlockStmt)
	if p.check.logosInPlace[lam] {
		return p.inPlace(lam, st)
	}
	if isBlock {
		return p.later(lam, body, st)
	}
	p.expr(lam.Body, p.lateCaptures(p.capture(st), lam, p.defsIn(lam)))
	return st
}

func (p *logosEFPass) inPlace(lam *syntax.LambdaExpr, st *logosEFState) *logosEFState {
	in := st
	for {
		var end *logosEFState
		if body, ok := lam.Body.(*syntax.BlockStmt); ok {
			end = p.run(body, in)
		} else {
			end = p.expr(lam.Body, in)
		}
		next := logosEFJoin(st, end)
		if logosEFSame(next, in) {
			return in
		}
		in = next
	}
}
