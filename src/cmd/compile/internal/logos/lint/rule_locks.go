// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// holdMode is how a lock is held at a program point.
type holdMode uint8

const (
	holdNone holdMode = iota
	holdRead
	holdWrite
)

// heldLock is one lock instance: the root variable or global and the dotted
// field path from it to the mutex. Path is a string so the type is
// comparable and can key a map.
type heldLock struct {
	Root   types2.Object
	Path   string
	Global bool
}

func (k heldLock) String() string {
	if k.Global || k.Root == nil {
		return k.Path
	}
	return k.Root.Name() + "." + k.Path
}

// heldSet maps each held lock to how it is held.
type heldSet map[heldLock]holdMode

func (h heldSet) clone() heldSet {
	out := make(heldSet, len(h))
	for k, v := range h {
		out[k] = v
	}
	return out
}

func (h heldSet) hold(k heldLock, m holdMode) {
	if m == holdNone {
		delete(h, k)
		return
	}
	if h[k] == holdWrite {
		return
	}
	h[k] = m
}

func (h heldSet) merge(o heldSet) {
	for k, m := range h {
		om, ok := o[k]
		switch {
		case !ok:
			delete(h, k)
		case om < m:
			h[k] = om
		}
	}
}

// lockState is the analysis state of one function body: the held locks and
// the defers registered so far in source order.
type lockState struct {
	held   heldSet
	defers []deferredCall
}

// deferredCall is one defer statement: an unlock, a contract call, or a
// closure body, checked at function exit in LIFO order.
type deferredCall struct {
	call   *syntax.CallExpr
	fnBody syntax.Node
	file   *syntax.File
}

// lockAudit is the per-package lock analysis.
type lockAudit struct {
	in        *Input
	dirs      *packageDirectives
	guards    map[types2.Object]guardPath
	atomics   map[types2.Object]bool
	contracts map[*syntax.FuncDecl]*funcContract
	byObject  map[types2.Object]*funcContract
	indexIDs  map[types2.Object]string

	unguarded []Diagnostic
	nonatomic []Diagnostic
	unmet     []Diagnostic
}

// LockFacts returns one locks record per contract effect of the package,
// spelled the way the facts sidecar and its consumers spell them.
func LockFacts(in *Input) []facts.LockFact {
	dirs := scanDirectives(in)
	var out []facts.LockFact
	for i := range dirs.Contracts {
		ct := &dirs.Contracts[i]
		obj, ok := ct.Object.(*types2.Func)
		if !ok {
			continue
		}
		name := facts.ObjectName(obj)
		if !facts.ValidObject(name) {
			continue
		}
		for _, e := range ct.Effects {
			out = append(out, facts.LockFact{Object: name, Effect: e.Effect, Path: strings.Join(e.Path.steps, ".")})
		}
	}
	return out
}

// analyzeLocks runs the three lock rules once per Pass and caches the
// findings for the rules to report.
func analyzeLocks(pass *Pass) *lockAudit {
	if pass.locks != nil {
		return pass.locks
	}
	c := &lockAudit{
		in:        pass.in,
		dirs:      scanDirectives(pass.in),
		guards:    make(map[types2.Object]guardPath),
		atomics:   make(map[types2.Object]bool),
		contracts: make(map[*syntax.FuncDecl]*funcContract),
		byObject:  make(map[types2.Object]*funcContract),
	}
	for _, g := range c.dirs.Guards {
		c.guards[g.Object] = g.Guard
	}
	for _, a := range c.dirs.Atomics {
		c.atomics[a.Object] = true
	}
	for i := range c.dirs.Contracts {
		ct := &c.dirs.Contracts[i]
		c.contracts[ct.Decl] = ct
		c.byObject[ct.Object] = ct
	}
	for _, r := range c.dirs.Misplaced {
		switch r.Rule {
		case guardedByRule:
			c.unguarded = append(c.unguarded, diagOf(r))
		case atomicRule:
			c.nonatomic = append(c.nonatomic, diagOf(r))
		case lockRequireRule:
			c.unmet = append(c.unmet, diagOf(r))
		}
	}
	for _, f := range pass.in.Files {
		c.checkFile(pass, f)
	}
	pass.locks = c
	return c
}

func diagOf(r Result) Diagnostic {
	return Diagnostic{Pos: r.Pos, End: r.End, Message: r.Message}
}

// checkFile analyzes every function declaration of f.
func (c *lockAudit) checkFile(pass *Pass, f *syntax.File) {
	for _, d := range f.DeclList {
		if fn, ok := d.(*syntax.FuncDecl); ok && fn.Body != nil {
			c.checkFunc(pass, fn, f)
		}
	}
}

// checkFunc walks one function body with the entry state its contract
// declares.
func (c *lockAudit) checkFunc(pass *Pass, decl *syntax.FuncDecl, f *syntax.File) {
	v := &lockFlow{c: c, pass: pass, decl: decl, file: f, methods: make(map[types2.Object]methodLock)}
	v.restart(pass, decl, f)
	v.solveBody(decl.Body)
}

func (v *lockFlow) restart(pass *Pass, decl *syntax.FuncDecl, f *syntax.File) {
	st := lockState{held: heldSet{}}
	if ct := v.c.contracts[decl]; ct != nil {
		v.c.applyContractEffects(pass, decl, f, ct, &st, syntax.Pos{})
		for _, eff := range ct.Effects {
			if eff.Effect != "requires" {
				continue
			}
			if k, ok := v.c.contractLockKey(decl, eff.Path); ok {
				st.held.hold(k, holdWrite)
			}
		}
	}
	v.st = &st
	v.local = make(map[types2.Object]bool)
	v.published = make(map[types2.Object]bool)
	v.labels = nil
	v.probe = false
}

func bodyHasLocalGoto(body *syntax.BlockStmt) bool {
	found := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.BranchStmt:
			found = found || n.Tok == syntax.Goto
		case *syntax.LabeledStmt:
			found = true
		}
		return true
	})
	return found
}

// lockFlow walks one function body. The body of a go statement, a group
// child and a function literal starts with no lock held, except a lambda
// or function literal argument of a chain method that is not a segment of
// a group stage: it runs in place with the caller's locks. The function
// and the arguments of a go statement are evaluated with the caller's
// locks.
type lockFlow struct {
	c         *lockAudit
	pass      *Pass
	decl      *syntax.FuncDecl
	file      *syntax.File
	st        *lockState
	local     map[types2.Object]bool
	published map[types2.Object]bool
	gotos     map[string]heldSet
	labels    *lockLabels
	methods   map[types2.Object]methodLock
	probe     bool
}

type lockLabels struct {
	bodies  map[*syntax.BlockStmt]map[string]heldSet
	changed bool
}

type methodLock struct {
	key heldLock
	op  string
}

func (v *lockFlow) walkStmts(list []syntax.Stmt) {
	for _, s := range list {
		if !v.walkStmt(s) {
			return
		}
	}
}

// walkStmt walks one statement and reports whether control continues after
// it.
func (v *lockFlow) walkStmt(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		v.walkStmts(s.List)
	case *syntax.LabeledStmt:
		v.mergeGoto(s.Label.Value)
		v.walkStmt(s.Stmt)
	case *syntax.BranchStmt:
		if s.Tok == syntax.Goto && s.Label != nil {
			v.noteGoto(s.Label.Value)
		}
	case *syntax.IfStmt:
		v.walkIf(s)
	case *syntax.ForStmt:
		v.walkFor(s)
	case *syntax.SwitchStmt:
		v.walkSwitch(s)
	case *syntax.SelectStmt:
		v.walkSelect(s)
	case *syntax.CallStmt:
		v.walkCallStmt(s)
	case *syntax.GroupStmt:
		v.walkGroup(s)
	case *syntax.GroupChild:
		v.walkChild(s)
	case *syntax.ReturnStmt:
		v.walkExpr(s.Results)
	case *syntax.AssignStmt:
		v.walkAssign(s)
	case *syntax.DeclStmt:
		v.walkDecls(s.DeclList)
	case *syntax.ExprStmt:
		v.walkExpr(s.X)
	case *syntax.SendStmt:
		v.walkExpr(s.Chan)
		v.walkExpr(s.Value)
	case syntax.SimpleStmt:
		v.walkSimple(s)
	}
	return true
}

func (v *lockFlow) noteGoto(label string) {
	if !v.probe || v.gotos == nil {
		return
	}
	if cur, ok := v.gotos[label]; ok {
		before := cur.clone()
		cur.merge(v.st.held)
		v.labels.changed = v.labels.changed || !maps.Equal(before, cur)
		return
	}
	v.gotos[label] = v.st.held.clone()
	v.labels.changed = true
}
func (v *lockFlow) mergeGoto(label string) {
	if v.gotos == nil {
		return
	}
	g, ok := v.gotos[label]
	if !ok {
		return
	}
	v.st.held.merge(g)
}

func (v *lockFlow) walkIf(s *syntax.IfStmt) {
	v.walkSimple(s.Init)
	v.walkExpr(s.Cond)
	before := v.st.held.clone()
	if key, ok := v.tryLockThen(s.Cond); ok {
		v.st.held.hold(key, holdWrite)
	}
	v.walkStmts(s.Then.List)
	afterThen := v.st.held
	v.st.held = before
	if s.Else != nil {
		if key, ok := v.tryLockElse(s.Cond); ok {
			v.st.held.hold(key, holdWrite)
		}
		v.walkStmt(s.Else)
	}
	afterElse := v.st.held
	v.st.held = afterThen
	v.st.held.merge(afterElse)
}

func (v *lockFlow) tryLockThen(cond syntax.Expr) (heldLock, bool) {
	return v.tryLockOperand(syntax.Unparen(cond))
}

func (v *lockFlow) tryLockElse(cond syntax.Expr) (heldLock, bool) {
	op, ok := syntax.Unparen(cond).(*syntax.Operation)
	if !ok || op.Op != syntax.Not || op.Y != nil {
		return heldLock{}, false
	}
	return v.tryLockOperand(op.X)
}

func (v *lockFlow) tryLockOperand(e syntax.Expr) (heldLock, bool) {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok || len(call.ArgList) != 0 {
		return heldLock{}, false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "TryLock" {
		return heldLock{}, false
	}
	key, _, ok := v.builtinLockOp(call)
	return key, ok
}

func (v *lockFlow) walkFor(s *syntax.ForStmt) {
	v.walkSimple(s.Init)
	v.walkExpr(s.Cond)
	before := v.st.held.clone()
	v.walkStmts(s.Body.List)
	v.walkSimple(s.Post)
	v.st.held.merge(before)
}

func (v *lockFlow) walkSwitch(s *syntax.SwitchStmt) {
	v.walkSimple(s.Init)
	v.walkExpr(s.Tag)
	entry := v.st.held.clone()
	var merged heldSet
	first := true
	for _, clause := range s.Body {
		v.st.held = entry.clone()
		v.walkExpr(clause.Cases)
		v.walkStmts(clause.Body)
		if first {
			merged = v.st.held.clone()
			first = false
			continue
		}
		merged.merge(v.st.held)
	}
	if first {
		merged = entry
	} else {
		merged.merge(entry)
	}
	v.st.held = merged
}

func (v *lockFlow) walkSelect(s *syntax.SelectStmt) {
	entry := v.st.held.clone()
	var merged heldSet
	first := true
	for _, clause := range s.Body {
		v.st.held = entry.clone()
		v.walkStmts(clause.Body)
		if first {
			merged = v.st.held.clone()
			first = false
			continue
		}
		merged.merge(v.st.held)
	}
	if first {
		merged = entry
	} else {
		merged.merge(entry)
	}
	v.st.held = merged
}

func (v *lockFlow) walkCallStmt(s *syntax.CallStmt) {
	call, _ := s.Call.(*syntax.CallExpr)
	if call == nil {
		v.walkExpr(s.Call)
		return
	}
	switch s.Tok.String() {
	case "defer":
		v.recordDefer(call)
		v.walkDeferArgs(call)
	case "go":
		v.walkSpawn(call)
	default:
		v.walkExpr(call)
	}
}

func (v *lockFlow) walkSpawn(call *syntax.CallExpr) {
	if body := literalBody(call); body != nil {
		v.walkInlineCall(call, body, heldSet{})
		return
	}
	v.walkOperands(call, nil)
	v.child(heldSet{}).applyCall(call)
}

func (v *lockFlow) walkGroup(s *syntax.GroupStmt) {
	v.walkExpr(s.Ctx)
	v.walkExpr(s.Limit)
	v.walkStmts(s.Body.List)
}

func (v *lockFlow) walkChild(s *syntax.GroupChild) {
	v.noteClosureCapture(s.Body)
	v.child(heldSet{}).walkBody(s.Body)
}

func (v *lockFlow) walkAssign(s *syntax.AssignStmt) {
	if s.Rhs == nil {
		v.walkLhs(s.Lhs)
		return
	}
	v.walkExpr(s.Rhs)
	if s.Op == syntax.Def {
		v.noteDefAllocs(s)
	}
	v.noteMethodValue(s)
	for _, lhs := range exprList(s.Lhs) {
		v.publishFrom(lhs, s.Rhs)
	}
	v.walkLhs(s.Lhs)
}

func (v *lockFlow) noteDefAllocs(s *syntax.AssignStmt) {
	allocs := allocValues(s.Rhs)
	for i, e := range exprList(s.Lhs) {
		if i >= len(allocs) || !allocs[i] {
			continue
		}
		name, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok {
			continue
		}
		if obj := v.c.in.defObject(name); obj != nil {
			v.local[obj] = true
		}
	}
}

func (v *lockFlow) noteMethodValue(s *syntax.AssignStmt) {
	lhs, rhs := exprList(s.Lhs), exprList(s.Rhs)
	if len(lhs) != 1 || len(rhs) != 1 {
		for _, el := range lhs {
			if name, ok := syntax.Unparen(el).(*syntax.Name); ok {
				if obj := v.methodObj(s, name); obj != nil {
					delete(v.methods, obj)
				}
			}
		}
		return
	}
	name, ok := syntax.Unparen(lhs[0]).(*syntax.Name)
	if !ok || isBlank(name) {
		return
	}
	obj := v.methodObj(s, name)
	if obj == nil {
		return
	}
	if m, ok := v.methodLockOf(rhs[0]); ok {
		v.methods[obj] = m
		return
	}
	delete(v.methods, obj)
}

func (v *lockFlow) methodObj(s *syntax.AssignStmt, name *syntax.Name) types2.Object {
	if s.Op == syntax.Def {
		return v.c.in.defObject(name)
	}
	return v.c.in.useObject(name)
}

func (v *lockFlow) methodLockOf(rhs syntax.Expr) (methodLock, bool) {
	sel, ok := syntax.Unparen(rhs).(*syntax.SelectorExpr)
	if !ok {
		return methodLock{}, false
	}
	selObj := v.c.selectionOf(sel)
	if selObj == nil || selObj.Kind() != types2.MethodVal {
		return methodLock{}, false
	}
	key, ok := v.c.keyOf(sel.X)
	if !ok {
		return methodLock{}, false
	}
	if !isMutexType(selObj.Recv()) {
		path, ok := v.c.promotedMutexPath(selObj, sel.X)
		if !ok {
			return methodLock{}, false
		}
		key, _ = joinContractKey(key, true, path)
	}
	switch sel.Sel.Value {
	case "Lock", "RLock", "Unlock", "RUnlock", "TryLock":
		return methodLock{key: key, op: sel.Sel.Value}, true
	}
	return methodLock{}, false
}

func (v *lockFlow) applyMethodLock(m methodLock) {
	switch m.op {
	case "Lock":
		v.st.held.hold(m.key, holdWrite)
	case "RLock":
		v.st.held.hold(m.key, holdRead)
	default:
		v.st.held.hold(m.key, holdNone)
	}
}
func (v *lockFlow) walkLhs(lhs syntax.Expr) {
	for _, e := range exprList(lhs) {
		switch x := syntax.Unparen(e).(type) {
		case *syntax.SelectorExpr:
			v.checkSelector(x, selWrite)
			v.walkExpr(x.X)
		case *syntax.Name:
			v.checkName(x, selWrite)
		default:
			v.walkExpr(e)
		}
	}
}

func (v *lockFlow) walkDecls(decls []syntax.Decl) {
	for _, d := range decls {
		if vd, ok := d.(*syntax.VarDecl); ok && vd.Values != nil {
			v.walkExpr(vd.Values)
			v.noteLocalAllocs(vd)
			v.noteMethodVar(vd)
		}
	}
}

func (v *lockFlow) noteMethodVar(vd *syntax.VarDecl) {
	vals := exprList(vd.Values)
	for i, nm := range vd.NameList {
		if i >= len(vals) {
			break
		}
		obj := v.c.in.defObject(nm)
		if obj == nil {
			continue
		}
		if m, ok := v.methodLockOf(vals[i]); ok {
			v.methods[obj] = m
			continue
		}
		delete(v.methods, obj)
	}
}

// walkSimple walks a simple statement that carries no control flow.
func (v *lockFlow) walkSimple(s syntax.SimpleStmt) {
	switch s := s.(type) {
	case nil:
	case *syntax.AssignStmt:
		v.walkAssign(s)
	case *syntax.ExprStmt:
		v.walkExpr(s.X)
	case *syntax.SendStmt:
		v.walkExpr(s.Chan)
		v.walkExpr(s.Value)
	case *syntax.RangeClause:
		v.walkExpr(s.X)
	}
}

// walkExpr walks an expression, flagging guarded and atomic field accesses
// and applying the lock effects of calls in evaluation order.
func (v *lockFlow) walkExpr(e syntax.Expr) {
	if e == nil {
		return
	}
	switch e := e.(type) {
	case *syntax.CallExpr:
		v.walkCall(e)
	case *syntax.SelectorExpr:
		v.checkSelector(e, selRead)
		v.walkExpr(e.X)
	case *syntax.Name:
		v.checkName(e, selRead)
	case *syntax.Operation:
		if e.Y == nil && e.Op == syntax.And {
			if sel, ok := syntax.Unparen(e.X).(*syntax.SelectorExpr); ok {
				v.checkAddress(sel)
				v.walkExpr(sel.X)
				v.checkAddressOf(sel)
			} else {
				v.walkExpr(e.X)
			}
		} else {
			v.walkExpr(e.X)
			v.walkExpr(e.Y)
		}
	case *syntax.ParenExpr:
		v.walkExpr(e.X)
	case *syntax.IndexExpr:
		v.walkExpr(e.X)
		v.walkExpr(e.Index)
	case *syntax.SliceExpr:
		v.walkExpr(e.X)
		for _, idx := range e.Index {
			v.walkExpr(idx)
		}
	case *syntax.AssertExpr:
		v.walkExpr(e.X)
	case *syntax.FuncLit:
		v.noteClosureCapture(e.Body)
		inner := v.child(heldSet{})
		inner.walkStmts(e.Body.List)
	case *syntax.LambdaExpr:
		v.noteClosureCapture(e.Body)
		inner := v.child(heldSet{})
		switch b := e.Body.(type) {
		case *syntax.BlockStmt:
			inner.walkStmts(b.List)
		case syntax.Expr:
			inner.walkExpr(b)
		}
	case *syntax.CompositeLit:
		for _, el := range e.ElemList {
			v.walkExpr(el)
		}
	case *syntax.KeyValueExpr:
		v.walkExpr(e.Key)
		v.walkExpr(e.Value)
	case *syntax.ListExpr:
		for _, el := range e.ElemList {
			v.walkExpr(el)
		}
	}
}

// walkCall applies the lock effects of call and walks its operands.
func (v *lockFlow) walkCall(call *syntax.CallExpr) {
	if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok && len(call.ArgList) == 0 {
		if obj := v.c.in.useObject(name); obj != nil {
			if m, ok := v.methods[obj]; ok {
				v.applyMethodLock(m)
				return
			}
		}
	}
	if body := literalBody(call); body != nil {
		v.walkInlineCall(call, body, v.st.held)
		return
	}
	inPlace := chainFuncs(v.pass, call)
	if stageOf(v.pass, call) != nil {
		inPlace = nil
	}
	v.walkOperands(call, inPlace)
	v.applyCall(call)
}

func (v *lockFlow) walkOperands(call *syntax.CallExpr, inPlace []syntax.Expr) {
	v.walkExpr(call.Fun)
	atomic := atomicQualifiedCall(v.pass, call)
	for _, arg := range call.ArgList {
		switch {
		case atomic:
			v.walkAtomicArg(arg)
		case slices.Contains(inPlace, syntax.Unparen(arg)):
			v.walkInPlace(syntax.Unparen(arg))
		default:
			v.walkExpr(arg)
		}
	}
}

func (v *lockFlow) walkInPlace(fn syntax.Expr) {
	body := callbackBody(fn)
	held := v.st.held.clone()
	v.noteClosureCapture(body)
	v.child(held).walkBody(body)
	v.st.held.merge(held)
}

func (v *lockFlow) applyCall(call *syntax.CallExpr) {
	if key, mode, ok := v.builtinLockOp(call); ok {
		switch mode {
		case holdWrite:
			v.st.held.hold(key, holdWrite)
		case holdRead:
			v.st.held.hold(key, holdRead)
		default:
			v.st.held.hold(key, holdNone)
		}
		return
	}
	if ct := v.callContract(call); ct != nil {
		v.applyCallContract(call, ct)
	}
	v.publishCallArgs(call)
}

func (v *lockFlow) walkInlineCall(call *syntax.CallExpr, body syntax.Node, held heldSet) {
	for _, arg := range call.ArgList {
		v.walkExpr(arg)
	}
	v.noteClosureCapture(body)
	v.child(held).walkBody(body)
	v.publishCallArgs(call)
}

func (v *lockFlow) child(held heldSet) *lockFlow {
	return &lockFlow{c: v.c, pass: v.pass, decl: v.decl, file: v.file, st: &lockState{held: held}, local: v.local, published: v.published, methods: v.methods, probe: v.probe, labels: v.labels}
}

func (v *lockFlow) solveBody(body *syntax.BlockStmt) {
	if !bodyHasLocalGoto(body) {
		v.walkStmts(body.List)
		v.checkDefers(body.List)
		return
	}
	v.labels = &lockLabels{bodies: map[*syntax.BlockStmt]map[string]heldSet{body: make(map[string]heldSet)}}
	for {
		v.labels.changed = false
		inner := v.child(v.st.held.clone())
		inner.local = maps.Clone(v.local)
		inner.published = maps.Clone(v.published)
		inner.methods = maps.Clone(v.methods)
		inner.probe = true
		inner.walkBody(body)
		if !v.labels.changed {
			break
		}
	}
	v.walkBody(body)
}

func (v *lockFlow) walkBody(body syntax.Node) {
	switch b := body.(type) {
	case *syntax.BlockStmt:
		if v.labels == nil {
			v.solveBody(b)
			return
		}
		gotos, seen := v.labels.bodies[b]
		if !seen {
			if bodyHasLocalGoto(b) {
				gotos = make(map[string]heldSet)
			}
			v.labels.bodies[b] = gotos
		}
		v.gotos = gotos
		v.walkStmts(b.List)
		v.checkDefers(b.List)
	case syntax.Expr:
		v.walkExpr(b)
	}
}

func (v *lockFlow) publishCallArgs(call *syntax.CallExpr) {
	for _, arg := range call.ArgList {
		if name, ok := syntax.Unparen(arg).(*syntax.Name); ok {
			if obj := v.c.in.useObject(name); obj != nil {
				v.published[obj] = true
			}
		}
	}
}

func (v *lockFlow) walkAtomicArg(arg syntax.Expr) {
	op, ok := syntax.Unparen(arg).(*syntax.Operation)
	if !ok || op.Op != syntax.And || op.Y != nil {
		v.walkExpr(arg)
		return
	}
	switch x := syntax.Unparen(op.X).(type) {
	case *syntax.SelectorExpr:
		v.checkAddress(x)
		v.walkExpr(x.X)
	case *syntax.Name:
	default:
		v.walkExpr(op.X)
	}
}

func atomicQualifiedCall(pass *Pass, call *syntax.CallExpr) bool {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	if pass.in.Info != nil && pass.in.Info.Selections[sel] != nil {
		return false
	}
	name, ok := syntax.Unparen(sel.X).(*syntax.Name)
	if !ok || pass.in.Info == nil {
		return false
	}
	pkg, ok := pass.in.Info.Uses[name].(*types2.PkgName)
	return ok && pkg.Imported() != nil && pkg.Imported().Path() == atomicPath
}

// builtinLockOp recognizes Lock, Unlock, RLock, RUnlock, and TryLock on a
// sync.Mutex or sync.RWMutex value.
func (v *lockFlow) builtinLockOp(call *syntax.CallExpr) (heldLock, holdMode, bool) {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return heldLock{}, holdNone, false
	}
	selObj := v.c.selectionOf(sel)
	if selObj == nil || selObj.Kind() != types2.MethodVal {
		return heldLock{}, holdNone, false
	}
	key, ok := v.c.keyOf(sel.X)
	if !ok {
		return heldLock{}, holdNone, false
	}
	if !isMutexType(selObj.Recv()) {
		path, ok := v.c.promotedMutexPath(selObj, sel.X)
		if !ok {
			return heldLock{}, holdNone, false
		}
		key, _ = joinContractKey(key, true, path)
	}
	switch sel.Sel.Value {
	case "Lock":
		return key, holdWrite, true
	case "RLock":
		return key, holdRead, true
	case "Unlock", "RUnlock":
		return key, holdNone, true
	case "TryLock":
		return key, holdNone, true
	}
	return heldLock{}, holdNone, false
}

func (c *lockAudit) promotedMutexPath(sel *types2.Selection, recv syntax.Expr) (string, bool) {
	if sel == nil || len(sel.Index()) < 2 {
		return "", false
	}
	t := TypeOf(recv)
	var steps []string
	for _, idx := range sel.Index()[:len(sel.Index())-1] {
		st, ok := structOf(t)
		if !ok || idx < 0 || idx >= st.NumFields() {
			return "", false
		}
		steps = append(steps, st.Field(idx).Name())
		t = st.Field(idx).Type()
	}
	if !isMutexType(t) {
		return "", false
	}
	return strings.Join(steps, "."), true
}

// callContract returns the declared contract of the function call invokes,
// from the package directives or the imported facts.
func (v *lockFlow) callContract(call *syntax.CallExpr) *funcContract {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if ok {
		selObj := v.c.selectionOf(sel)
		if selObj == nil || selObj.Kind() != types2.MethodVal {
			return nil
		}
		fn, ok := selObj.Obj().(*types2.Func)
		if !ok {
			return nil
		}
		return v.c.contractFor(fn)
	}
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return nil
	}
	fn, ok := v.c.in.useObject(name).(*types2.Func)
	if !ok {
		return nil
	}
	return v.c.contractFor(fn)
}

// contractFor returns the lock contract of fn, from the same package's
// directives or from the Locks facts of an imported package.
func (c *lockAudit) contractFor(fn *types2.Func) *funcContract {
	if ct := c.byObject[fn]; ct != nil {
		return ct
	}
	obj := facts.ObjectName(fn)
	if obj == "" || fn.Pkg() == nil {
		return nil
	}
	f, ok := facts.Lookup(fn.Pkg().Path())
	if !ok || f == nil {
		return nil
	}
	var ct *funcContract
	for _, l := range f.Locks {
		if l.Object != obj {
			continue
		}
		path, ok := parseGuardPath(l.Path)
		if !ok {
			continue
		}
		if ct == nil {
			ct = &funcContract{}
		}
		ct.Effects = append(ct.Effects, contractEffect{Effect: l.Effect, Path: path})
	}
	return ct
}

// applyCallContract binds the contract paths of the callee to the operands
// of call and checks and applies the effects.
func (v *lockFlow) applyCallContract(call *syntax.CallExpr, ct *funcContract) {
	fn := v.calledFunc(call)
	for _, eff := range ct.Effects {
		key, ok := v.c.callLockKey(fn, call, eff.Path)
		if !ok {
			continue
		}
		switch eff.Effect {
		case "requires":
			if v.st.held[key] == holdNone && !v.probe {
				v.c.reportUnmet(v.decl, fn, eff.Path, call.Pos())
			}
		case "acquires":
			v.st.held.hold(key, holdWrite)
		case "releases":
			v.st.held.hold(key, holdNone)
		}
	}
}

// calledFunc returns the function call invokes, or nil.
func (v *lockFlow) calledFunc(call *syntax.CallExpr) *types2.Func {
	if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok {
		if selObj := v.c.selectionOf(sel); selObj != nil {
			fn, _ := selObj.Obj().(*types2.Func)
			return fn
		}
	}
	if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok {
		fn, _ := v.c.in.useObject(name).(*types2.Func)
		return fn
	}
	return nil
}

// callLockKey binds the path of a callee contract, rooted at a parameter or
// the receiver name of fn, to the operands of call.
func (c *lockAudit) callLockKey(fn *types2.Func, call *syntax.CallExpr, path guardPath) (heldLock, bool) {
	if fn == nil || len(path.steps) == 0 {
		return heldLock{}, false
	}
	root := path.steps[0]
	rest := strings.Join(path.steps[1:], ".")
	if recvOperand, ok := c.receiverOperand(call); ok {
		if sig, ok := fn.Type().(*types2.Signature); ok && sig.Recv() != nil && sig.Recv().Name() == root {
			k, kok := c.keyOf(recvOperand)
			return joinContractKey(k, kok, rest)
		}
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok {
		return heldLock{}, false
	}
	params := sig.Params()
	args := call.ArgList
	for i := 0; i < params.Len() && i < len(args); i++ {
		if params.At(i).Name() != root {
			continue
		}
		k, kok := c.keyOf(args[i])
		return joinContractKey(k, kok, rest)
	}
	return heldLock{}, false
}

func joinContractKey(k heldLock, ok bool, rest string) (heldLock, bool) {
	if !ok || rest == "" {
		return k, ok
	}
	if k.Path == "" {
		k.Path = rest
	} else {
		k.Path += "." + rest
	}
	return k, true
}

// receiverOperand returns the receiver expression of a method call.
func (c *lockAudit) receiverOperand(call *syntax.CallExpr) (syntax.Expr, bool) {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return nil, false
	}
	if selObj := c.selectionOf(sel); selObj == nil || selObj.Kind() != types2.MethodVal {
		return nil, false
	}
	return sel.X, true
}

type selAccess uint8

const (
	selRead selAccess = iota
	selWrite
	selAddress
)

func (v *lockFlow) checkAddress(sel *syntax.SelectorExpr) { v.checkSelector(sel, selAddress) }

func (v *lockFlow) checkSelector(sel *syntax.SelectorExpr, access selAccess) {
	selObj := v.c.selectionOf(sel)
	if selObj == nil || selObj.Kind() != types2.FieldVal {
		return
	}
	obj := selObj.Obj()
	if v.c.atomics[obj] && access != selAddress {
		if !v.probe {
			v.c.reportNonatomic(v.decl, sel.Pos())
		}
		return
	}
	guard, ok := v.c.guards[obj]
	if !ok {
		return
	}
	if v.exempt(obj, sel.X) {
		return
	}
	key, ok := v.c.keyOfRootedGuard(sel.X, guard)
	if !ok {
		if !v.probe {
			v.c.reportUnguarded(v.decl, sel.Pos())
		}
		return
	}
	held := v.st.held[key]
	if held == holdNone || access == selWrite && held == holdRead {
		if !v.probe {
			v.c.reportUnguarded(v.decl, sel.Pos())
		}
	}
}

func (v *lockFlow) checkName(name *syntax.Name, access selAccess) {
	obj := v.c.in.useObject(name)
	if obj == nil {
		return
	}
	if v.c.atomics[obj] && access != selAddress {
		if !v.probe {
			v.c.reportNonatomic(v.decl, name.Pos())
		}
		return
	}
	guard, ok := v.c.guards[obj]
	if !ok {
		return
	}
	if v.exempt(obj, name) {
		return
	}
	key, ok := v.c.keyOfRootedGuard(name, guard)
	if !ok {
		if !v.probe {
			v.c.reportUnguarded(v.decl, name.Pos())
		}
		return
	}
	held := v.st.held[key]
	if held == holdNone || access == selWrite && held == holdRead {
		if !v.probe {
			v.c.reportUnguarded(v.decl, name.Pos())
		}
	}
}

// checkAddressOf flags address-of of an atomic field outside a sync/atomic
// call.
func (v *lockFlow) checkAddressOf(sel *syntax.SelectorExpr) {
	selObj := v.c.selectionOf(sel)
	if selObj == nil || selObj.Kind() != types2.FieldVal {
		return
	}
	if v.c.atomics[selObj.Obj()] && !v.probe {
		v.c.reportNonatomic(v.decl, sel.Pos())
	}
}

// exempt reports whether obj read through root is a local object allocated
// in this function and not yet published.
func (v *lockFlow) exempt(obj types2.Object, root syntax.Expr) bool {
	rootObj := v.rootObject(root)
	if rootObj == nil || !v.local[rootObj] || v.published[rootObj] {
		return false
	}
	return true
}

func (v *lockFlow) rootObject(e syntax.Expr) types2.Object {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		return v.c.in.useObject(e)
	case *syntax.SelectorExpr:
		return v.rootObject(e.X)
	case *syntax.IndexExpr:
		return v.rootObject(e.X)
	case *syntax.Operation:
		return v.rootObject(e.X)
	}
	return nil
}

// noteLocalAllocs records the variables of vd whose value is a fresh object
// allocated here.
func (v *lockFlow) noteLocalAllocs(vd *syntax.VarDecl) {
	allocs := allocValues(vd.Values)
	for i, name := range vd.NameList {
		if i >= len(allocs) {
			break
		}
		if !allocs[i] {
			continue
		}
		if obj := v.c.in.defObject(name); obj != nil {
			v.local[obj] = true
		}
	}
}

// allocValues reports, per value of a variable declaration, whether the
// value is a fresh allocation of this function.
func allocValues(values syntax.Expr) []bool {
	list := exprList(values)
	out := make([]bool, len(list))
	for i, e := range list {
		out[i] = isFreshAlloc(e)
	}
	return out
}

// isFreshAlloc reports whether e builds a new object: &T{}, new(T), or a
// composite literal whose type is a pointer.
func isFreshAlloc(e syntax.Expr) bool {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Operation:
		if e.Op != syntax.And || e.Y != nil {
			return false
		}
		_, ok := syntax.Unparen(e.X).(*syntax.CompositeLit)
		return ok
	case *syntax.CallExpr:
		return false
	case *syntax.CompositeLit:
		return false
	}
	return false
}

// publishFrom marks roots published by an assignment of rhs to lhs.
func (v *lockFlow) publishFrom(lhs, rhs syntax.Expr) {
	for _, r := range exprList(rhs) {
		root := v.rootObject(r)
		if root == nil {
			continue
		}
		lhsRoot := v.rootObject(lhs)
		if lhsRoot == root {
			continue
		}
		v.published[root] = true
	}
}

func (v *lockFlow) noteClosureCapture(body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		name, ok := n.(*syntax.Name)
		if !ok {
			return true
		}
		obj := v.c.in.useObject(name)
		if obj == nil {
			return true
		}
		if v.local[obj] {
			v.published[obj] = true
		}
		return true
	})
}

func literalBody(call *syntax.CallExpr) syntax.Node {
	switch f := syntax.Unparen(call.Fun).(type) {
	case *syntax.FuncLit:
		return f.Body
	case *syntax.LambdaExpr:
		return f.Body
	}
	return nil
}

// recordDefer registers one defer statement.
func (v *lockFlow) recordDefer(call *syntax.CallExpr) {
	d := deferredCall{call: call, file: v.file, fnBody: literalBody(call)}
	v.st.defers = append(v.st.defers, d)
}

// walkDeferArgs walks the operands a defer statement evaluates now.
func (v *lockFlow) walkDeferArgs(call *syntax.CallExpr) {
	switch syntax.Unparen(call.Fun).(type) {
	case *syntax.FuncLit, *syntax.LambdaExpr:
	default:
		v.walkExpr(call.Fun)
	}
	for _, arg := range call.ArgList {
		v.walkExpr(arg)
	}
}

// checkDefers walks the deferred calls registered in list at function exit
// in LIFO order against the held set each exit path leaves. A lock whose
// unlock is deferred stays held through the body (the defer records it) and
// only leaves the set at its turn in the LIFO walk.
func (v *lockFlow) checkDefers(list []syntax.Stmt) {
	exit := v.exitHeld()
	defers := v.deferredInOrder(list)
	for i := len(defers) - 1; i >= 0; i-- {
		d := defers[i]
		if d.fnBody != nil {
			inner := v.child(exit.clone())
			inner.file = d.file
			switch b := d.fnBody.(type) {
			case *syntax.BlockStmt:
				inner.walkStmts(b.List)
			case syntax.Expr:
				inner.walkExpr(b)
			}
			continue
		}
		if key, mode, ok := v.builtinLockOp(d.call); ok {
			exit.hold(key, mode)
			continue
		}
		if ct := v.callContract(d.call); ct != nil {
			sub := v.child(exit)
			sub.file = d.file
			sub.applyCallContract(d.call, ct)
		}
	}
	clear(v.st.held)
	maps.Copy(v.st.held, exit)
}

// exitHeld returns the held set at function exit: the walk's final set,
// since every return path merged by the statement walk.
func (v *lockFlow) exitHeld() heldSet { return v.st.held.clone() }

// deferredInOrder returns the defer statements of the body in source order.
func (v *lockFlow) deferredInOrder(list []syntax.Stmt) []deferredCall {
	var out []deferredCall
	var walk func(stmts []syntax.Stmt)
	walk = func(stmts []syntax.Stmt) {
		for _, s := range stmts {
			switch s := s.(type) {
			case *syntax.CallStmt:
				if call, ok := s.Call.(*syntax.CallExpr); ok && s.Tok.String() == "defer" {
					d := deferredCall{call: call, file: v.file, fnBody: literalBody(call)}
					out = append(out, d)
				}
			case *syntax.BlockStmt:
				walk(s.List)
			case *syntax.IfStmt:
				walk(s.Then.List)
				if s.Else != nil {
					if b, ok := s.Else.(*syntax.BlockStmt); ok {
						walk(b.List)
					}
				}
			case *syntax.ForStmt:
				walk(s.Body.List)
			}
		}
	}
	walk(list)
	return out
}

func (c *lockAudit) selectionOf(e *syntax.SelectorExpr) *types2.Selection {
	if c.in.Info == nil || c.in.Info.Selections == nil {
		return nil
	}
	return c.in.Info.Selections[e]
}

// keyOf returns the lock key of a lock expression: a variable, a global, or
// a field chain from one.
func (c *lockAudit) keyOf(e syntax.Expr) (heldLock, bool) {
	steps, root, ok := c.pathSteps(e)
	if !ok {
		return heldLock{}, false
	}
	if name, isName := root.(*syntax.Name); isName {
		if obj := c.in.useObject(name); obj != nil {
			if obj.Parent() == c.in.Pkg.Scope() {
				return heldLock{Root: obj, Path: strings.Join(steps, "."), Global: true}, true
			}
			return heldLock{Root: obj, Path: strings.Join(steps, ".")}, true
		}
	}
	return heldLock{}, false
}

// keyOfRootedGuard returns the lock key of a guard path reached from the
// expression root of a field access.
func (c *lockAudit) keyOfRootedGuard(rootExpr syntax.Expr, guard guardPath) (heldLock, bool) {
	if len(guard.steps) > 0 {
		if obj := c.in.Pkg.Scope().Lookup(guard.steps[0]); obj != nil {
			return heldLock{Root: obj, Path: strings.Join(guard.steps[1:], "."), Global: true}, true
		}
	}
	steps, root, ok := c.pathSteps(rootExpr)
	if !ok {
		return heldLock{}, false
	}
	all := append(slices.Clone(steps), guard.steps...)
	if name, isName := root.(*syntax.Name); isName {
		if obj := c.in.useObject(name); obj != nil {
			if obj.Parent() == c.in.Pkg.Scope() {
				return heldLock{Root: obj, Path: strings.Join(all, "."), Global: true}, true
			}
			return heldLock{Root: obj, Path: strings.Join(all, ".")}, true
		}
	}
	return heldLock{}, false
}

// contractLockKey returns the key of a requires path inside the callee body.
func (c *lockAudit) contractLockKey(decl *syntax.FuncDecl, path guardPath) (heldLock, bool) {
	if len(path.steps) == 0 {
		return heldLock{}, false
	}
	if obj := c.in.declParamObject(decl, path.steps[0]); obj != nil {
		return heldLock{Root: obj, Path: strings.Join(path.steps[1:], ".")}, true
	}
	return heldLock{}, false
}

// pathSteps returns the field steps and the root expression of e.
func (c *lockAudit) pathSteps(e syntax.Expr) ([]string, syntax.Expr, bool) {
	var steps []string
	for {
		switch x := syntax.Unparen(e).(type) {
		case *syntax.Name:
			slices.Reverse(steps)
			return steps, x, true
		case *syntax.Operation:
			e = x.X
		case *syntax.SelectorExpr:
			if sel := c.selectionOf(x); sel != nil && sel.Kind() != types2.FieldVal {
				return nil, nil, false
			}
			steps = append(steps, x.Sel.Value)
			e = x.X
		case *syntax.IndexExpr:
			step, ok := lockIndexStep(c.in.Info, &c.indexIDs, x)
			if !ok {
				return nil, nil, false
			}
			steps = append(steps, step)
			e = x.X
		default:
			return nil, nil, false
		}
	}
}

func lockIndexID(ids *map[types2.Object]string, obj types2.Object) string {
	if *ids == nil {
		*ids = make(map[types2.Object]string)
	}
	if id, ok := (*ids)[obj]; ok {
		return id
	}
	id := strings.Repeat("#", len(*ids)+1)
	(*ids)[obj] = id
	return id
}

func lockIndexStep(info *types2.Info, ids *map[types2.Object]string, x *syntax.IndexExpr) (string, bool) {
	if info == nil || info.Uses == nil {
		return "", false
	}
	switch i := syntax.Unparen(x.Index).(type) {
	case *syntax.BasicLit:
		if i.Bad {
			return "", false
		}
		return i.Value, true
	case *syntax.Name:
		obj, ok := info.Uses[i].(*types2.Var)
		if !ok || obj == nil {
			return "", false
		}
		return lockIndexID(ids, obj), true
	}
	return "", false
}

// applyContractEffects applies a same-package contract to st at a body or
// defer entry.
func (c *lockAudit) applyContractEffects(pass *Pass, decl *syntax.FuncDecl, f *syntax.File, ct *funcContract, st *lockState, at syntax.Pos) {
	for _, eff := range ct.Effects {
		key, ok := c.contractLockKey(decl, eff.Path)
		if !ok {
			continue
		}
		switch eff.Effect {
		case "acquires":
			st.held.hold(key, holdWrite)
		case "releases":
			st.held.hold(key, holdNone)
		}
	}
}

func (c *lockAudit) reportUnguarded(decl *syntax.FuncDecl, pos syntax.Pos) {
	c.unguarded = append(c.unguarded, Diagnostic{
		Pos:     pos,
		End:     pos,
		Message: fmt.Sprintf("%s: access without the guard", declName(decl)),
	})
}

func (c *lockAudit) reportNonatomic(decl *syntax.FuncDecl, pos syntax.Pos) {
	c.nonatomic = append(c.nonatomic, Diagnostic{
		Pos:     pos,
		End:     pos,
		Message: fmt.Sprintf("%s: plain access of an //logos:atomic field", declName(decl)),
	})
}

func (c *lockAudit) reportUnmet(decl *syntax.FuncDecl, callee *types2.Func, path guardPath, pos syntax.Pos) {
	name := "call"
	if callee != nil {
		name = facts.ObjectName(callee)
		if callee.Pkg() != nil && c.in.Pkg != nil && callee.Pkg() != c.in.Pkg {
			pkg := callee.Pkg().Name()
			if strings.HasPrefix(name, "(*") {
				name = "(*" + pkg + "." + name[2:]
			} else {
				name = pkg + "." + name
			}
		}
	}
	c.unmet = append(c.unmet, Diagnostic{
		Pos:     pos,
		End:     pos,
		Message: fmt.Sprintf("%s: call of %s requires %s", declName(decl), name, path.String()),
	})
}
