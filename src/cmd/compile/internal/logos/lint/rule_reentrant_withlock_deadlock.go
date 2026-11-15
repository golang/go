// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// logosSyncPath is the import path of the loGo/s synchronization package.
const logosSyncPath = "logos/sync"

// reentrantWithLockDeadlock is concurrency::reentrant_withlock_deadlock
// (SPEC Sec 1.2, #167): inside the function literal or lambda passed to
// WithLock of a logos/sync.Locked, a call to WithLock on the same receiver
// expression, which blocks forever on the mutex the outer call holds. The
// inner call counts when it appears directly in the callback, in the body
// of a directly invoked or deferred function literal, in a lambda or
// function literal argument of a chain method call or a named function
// (an explicit instantiation of a generic one included), method value, or
// stored literal passed to one (a plain chain runs it
// in place, and the stage of a group chain joins its children before the
// call returns), or anywhere in the synchronous call graph of the
// same-package functions the callback calls, at any depth. Each
// same-package function carries the set of receiver
// keys its synchronous execution can reach, composed across call sites
// through the existing lockKey substitution, so mutually recursive helpers
// reach a fixed point with no call-depth cap. A named same-package
// function or a method value passed as the callback is analyzed as the
// callback body with its receiver bound to the call-site operand. A go
// statement evaluates its receiver and arguments synchronously and only
// invokes its call asynchronously, so the invoked call and a literal that
// is its function are excluded while argument subexpressions stay in the
// flow. A stored function literal that is never invoked is skipped. Two
// receiver expressions are the same when they name the same variable
// through the same chain of field selections, ignoring address-of and
// dereference; a receiver that is any other expression never matches, so
// the rule fails safe. The analysis reads syntax and checker objects only,
// so it holds for Go and loGo/s packages alike; a lambda exists in loGo/s
// only.
var reentrantWithLockDeadlock = Rule{
	ID:    "concurrency::reentrant_withlock_deadlock",
	Group: Concurrency,
	Go:    true,
	Check: checkReentrantWithLock,
}

// lockKey names a receiver expression: the variable at its root and the
// chain of field names selected from it.
type lockKey struct {
	root types2.Object
	path string
}

// lockCall is one call of WithLock on a logos/sync.Locked, on the receiver
// expression recv. Body is the callback body when the argument is a
// function literal, a lambda, or a resolvable named callback, and nil
// otherwise. Bind maps callback-frame roots to caller-frame keys when the
// callback is a method value; cbDecl is the owning declaration of a named
// callback body, cbTarget its function, and cbFile is the file that owns
// body.
type lockCall struct {
	call     *syntax.CallExpr
	recv     syntax.Expr
	key      lockKey
	body     syntax.Node
	bind     map[types2.Object]lockKey
	cbDecl   *syntax.FuncDecl
	cbTarget *types2.Func
	cbFile   *syntax.File
}

// lockSummary is the relation R(fn) of receiver keys, in fn's own frame,
// with path length at most the check's keyPathLimit, reached by fn's
// synchronous execution. Queued records worklist membership.
type lockSummary struct {
	keys   map[lockKey]bool
	queued bool
}

// lockEdge is one synchronous call of callee by caller at call.
type lockEdge struct {
	caller *types2.Func
	call   *syntax.CallExpr
	callee *types2.Func
}

// lockCallbackEdge is one WithLock call by caller whose callback is the
// named same-package function target: the target body runs synchronously
// with the callback binding. Bind maps target-frame roots to caller-frame
// keys, and decl is the target declaration.
type lockCallbackEdge struct {
	caller *types2.Func
	bind   map[types2.Object]lockKey
	decl   *syntax.FuncDecl
}

// lockCheck is the analysis of one package. File is the file being
// checked.
type lockCheck struct {
	pass         *Pass
	file         *syntax.File
	calls        map[*syntax.CallExpr]lockCall
	callbacks    map[syntax.Node]bool
	decls        map[*types2.Func]*syntax.FuncDecl
	declFiles    map[*syntax.FuncDecl]*syntax.File
	summaries    map[*types2.Func]*lockSummary
	callers      map[*types2.Func][]lockEdge
	cbCallers    map[*types2.Func][]lockCallbackEdge
	fileOf       map[*syntax.CallExpr]*syntax.File
	asyncCall    map[*syntax.CallExpr]bool
	aliases      map[types2.Object]lockKey
	stored       map[types2.Object]syntax.Node
	indexIDs     map[types2.Object]string
	keyPathLimit int
	reported     map[*syntax.CallExpr]bool
	valueCalls   map[syntax.Expr]*syntax.CallExpr
}

// checkReentrantWithLock is the Check of reentrantWithLockDeadlock.
func checkReentrantWithLock(pass *Pass) {
	c := &lockCheck{
		pass:       pass,
		calls:      make(map[*syntax.CallExpr]lockCall),
		callbacks:  make(map[syntax.Node]bool),
		decls:      make(map[*types2.Func]*syntax.FuncDecl),
		declFiles:  make(map[*syntax.FuncDecl]*syntax.File),
		summaries:  make(map[*types2.Func]*lockSummary),
		callers:    make(map[*types2.Func][]lockEdge),
		cbCallers:  make(map[*types2.Func][]lockCallbackEdge),
		fileOf:     make(map[*syntax.CallExpr]*syntax.File),
		asyncCall:  make(map[*syntax.CallExpr]bool),
		reported:   make(map[*syntax.CallExpr]bool),
		valueCalls: make(map[syntax.Expr]*syntax.CallExpr),
	}
	for _, f := range pass.Files {
		c.markAsync(f)
		c.index(f)
		c.indexAliases(f)
		c.indexStored(f)
	}
	for _, f := range pass.Files {
		c.indexCalls(f)
	}
	c.buildSummaries()
	for _, f := range pass.Files {
		c.file = f
		v := &syncVisitor{c: c, file: f, funLit: make(map[syntax.Node]bool)}
		v.onCall = func(call *syntax.CallExpr) {
			c.checkOuter(call)
		}
		syntax.Walk(f, v)
	}
}

// origin returns the generic origin of fn, or fn itself.
func origin(fn *types2.Func) *types2.Func {
	if fn == nil {
		return nil
	}
	if o := fn.Origin(); o != nil {
		return o
	}
	return fn
}

// markAsync records the invoked call of every go statement: its body runs
// asynchronously, while its receiver and argument subexpressions still run
// synchronously.
func (c *lockCheck) markAsync(f *syntax.File) {
	syntax.Inspect(f, func(n syntax.Node) bool {
		if s, ok := n.(*syntax.CallStmt); ok && s.Tok == syntax.Go {
			if call := callOf(s.Call); call != nil {
				c.asyncCall[call] = true
			}
		}
		return true
	})
}

// index records the function declarations of f.
func (c *lockCheck) index(f *syntax.File) {
	for _, d := range f.DeclList {
		if fd, ok := d.(*syntax.FuncDecl); ok && fd.Body != nil {
			if fn, _ := c.pass.Info.Defs[fd.Name].(*types2.Func); fn != nil {
				c.decls[origin(fn)] = fd
				c.declFiles[fd] = f
			}
		}
	}
}

func (c *lockCheck) indexAliases(f *syntax.File) {
	counts := make(map[types2.Object]int)
	targets := make(map[types2.Object]syntax.Expr)
	syntax.Inspect(f, func(n syntax.Node) bool {
		switch x := n.(type) {
		case *syntax.AssignStmt:
			c.noteAlias(x, counts, targets)
		case *syntax.VarDecl:
			c.noteVarAlias(x, counts, targets)
		}
		return true
	})
	for obj, n := range counts {
		if n != 1 {
			continue
		}
		e, ok := targets[obj]
		if !ok {
			continue
		}
		if k := c.keyOf(e); k.root != nil {
			if c.aliases == nil {
				c.aliases = make(map[types2.Object]lockKey)
			}
			c.aliases[obj] = k
		}
	}
	for obj, k := range c.aliases {
		if r, ok := c.resolveAliasChain(k); ok {
			c.aliases[obj] = r
		} else {
			delete(c.aliases, obj)
		}
	}
}

func (c *lockCheck) resolveAliasChain(k lockKey) (lockKey, bool) {
	seen := make(map[types2.Object]bool)
	for {
		a, ok := c.aliases[k.root]
		if !ok {
			return k, true
		}
		if seen[k.root] {
			return lockKey{}, false
		}
		seen[k.root] = true
		k = lockKey{root: a.root, path: a.path + k.path}
	}
}

func (c *lockCheck) noteAlias(x *syntax.AssignStmt, counts map[types2.Object]int, targets map[types2.Object]syntax.Expr) {
	lhs, rhs := exprList(x.Lhs), exprList(x.Rhs)
	if len(lhs) != 1 || len(rhs) != 1 {
		return
	}
	name, ok := syntax.Unparen(lhs[0]).(*syntax.Name)
	if !ok || isBlank(name) {
		return
	}
	obj := c.pass.Info.Uses[name]
	if x.Op == syntax.Def {
		obj = c.pass.Info.Defs[name]
	}
	c.noteAliasTarget(obj, rhs[0], counts, targets)
}

func (c *lockCheck) noteVarAlias(x *syntax.VarDecl, counts map[types2.Object]int, targets map[types2.Object]syntax.Expr) {
	vals := exprList(x.Values)
	for i, nm := range x.NameList {
		if i >= len(vals) {
			break
		}
		c.noteAliasTarget(c.pass.Info.Defs[nm], vals[i], counts, targets)
	}
}

func (c *lockCheck) noteAliasTarget(obj types2.Object, rhs syntax.Expr, counts map[types2.Object]int, targets map[types2.Object]syntax.Expr) {
	if obj == nil {
		return
	}
	counts[obj]++
	if r := c.aliasTarget(rhs); r != nil {
		targets[obj] = r
	}
}

func (c *lockCheck) aliasTarget(rhs syntax.Expr) syntax.Expr {
	r := syntax.Unparen(rhs)
	if op, ok := r.(*syntax.Operation); ok && op.Op == syntax.And && op.Y == nil {
		return r
	}
	if name, ok := r.(*syntax.Name); ok && !isBlank(name) {
		if _, ok := TypeOf(rhs).(*types2.Pointer); ok {
			return r
		}
	}
	return nil
}

func (c *lockCheck) indexStored(f *syntax.File) {
	counts := make(map[types2.Object]int)
	lits := make(map[types2.Object]syntax.Node)
	syntax.Inspect(f, func(n syntax.Node) bool {
		switch x := n.(type) {
		case *syntax.AssignStmt:
			c.noteStored(x, counts, lits)
		case *syntax.VarDecl:
			c.noteStoredVar(x, counts, lits)
		}
		return true
	})
	if c.stored == nil {
		c.stored = make(map[types2.Object]syntax.Node)
	}
	for obj, n := range counts {
		if n != 1 {
			continue
		}
		if lit, ok := lits[obj]; ok {
			c.stored[obj] = lit
		}
	}
}

func (c *lockCheck) noteStored(x *syntax.AssignStmt, counts map[types2.Object]int, lits map[types2.Object]syntax.Node) {
	lhs, rhs := exprList(x.Lhs), exprList(x.Rhs)
	if len(lhs) != 1 || len(rhs) != 1 {
		return
	}
	name, ok := syntax.Unparen(lhs[0]).(*syntax.Name)
	if !ok || isBlank(name) {
		return
	}
	obj := c.pass.Info.Uses[name]
	if x.Op == syntax.Def {
		obj = c.pass.Info.Defs[name]
	}
	c.noteStoredBinding(obj, rhs[0], counts, lits)
}

func (c *lockCheck) noteStoredVar(x *syntax.VarDecl, counts map[types2.Object]int, lits map[types2.Object]syntax.Node) {
	vals := exprList(x.Values)
	for i, nm := range x.NameList {
		if i >= len(vals) {
			break
		}
		c.noteStoredBinding(c.pass.Info.Defs[nm], vals[i], counts, lits)
	}
}

func (c *lockCheck) noteStoredBinding(obj types2.Object, rhs syntax.Expr, counts map[types2.Object]int, lits map[types2.Object]syntax.Node) {
	if obj == nil {
		return
	}
	counts[obj]++
	if lit := immediateLiteral(rhs); lit != nil {
		lits[obj] = lit
	}
}

// indexCalls records the synchronous WithLock calls of f. Calls in
// asynchronous or stored bodies are excluded, so they are never indexed
// and never queried.
func (c *lockCheck) indexCalls(f *syntax.File) {
	v := &syncVisitor{c: c, file: f, funLit: make(map[syntax.Node]bool)}
	v.onCall = func(call *syntax.CallExpr) {
		c.noteLockCall(call, f)
	}
	syntax.Walk(f, v)
}

// noteLockCall records call when it calls WithLock on a logos/sync.Locked
// with a receiver expression the rule can match.
func (c *lockCheck) noteLockCall(call *syntax.CallExpr, f *syntax.File) {
	fn := calleeFunc(c.pass, call)
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if fn == nil || !ok || fn.Name() != "WithLock" || fn.Pkg() == nil || fn.Pkg().Path() != logosSyncPath {
		return
	}
	key := c.keyOf(sel.X)
	key.path += c.promotedPath(sel, sel.X)
	if key.root == nil {
		return
	}
	lc := lockCall{call: call, recv: sel.X, key: key, cbFile: f}
	c.fileOf[call] = f
	if len(call.ArgList) == 1 && !c.asyncCall[call] {
		arg := call.ArgList[0]
		if body := callbackBody(arg); body != nil {
			lc.body = body
			c.callbacks[syntax.Unparen(arg)] = true
		} else if target, recv := c.callbackTarget(arg); target != nil {
			if decl, ok := c.decls[origin(target)]; ok && decl.Body != nil {
				lc.body = decl.Body
				lc.cbDecl = decl
				lc.cbTarget = origin(target)
				lc.cbFile = c.declFiles[decl]
				if lc.cbFile == nil {
					lc.cbFile = f
				}
				if recv != nil && decl.Recv != nil {
					if robj := c.definedBy(decl.Recv); robj != nil {
						if rk := c.keyOf(recv); rk.root != nil {
							lc.bind = map[types2.Object]lockKey{robj: rk}
						}
					}
				}
			}
		}
	}
	c.calls[call] = lc
}

func (c *lockCheck) promotedPath(sel *syntax.SelectorExpr, recv syntax.Expr) string {
	s := c.pass.Info.Selections[sel]
	if s == nil || len(s.Index()) < 2 {
		return ""
	}
	t := TypeOf(recv)
	var sb strings.Builder
	for _, idx := range s.Index()[:len(s.Index())-1] {
		st, ok := structOf(t)
		if !ok || idx < 0 || idx >= st.NumFields() {
			return ""
		}
		sb.WriteString("." + st.Field(idx).Name())
		t = st.Field(idx).Type()
	}
	return sb.String()
}

// callbackTarget resolves a WithLock callback argument that is a plain
// function name, an explicit instantiation of a generic function, or a
// method value to its function, returning the method value receiver for
// the last and nil for the others. It returns nil when the argument has
// no resolvable same-package body.
func (c *lockCheck) callbackTarget(arg syntax.Expr) (*types2.Func, syntax.Expr) {
	switch x := syntax.Unparen(arg).(type) {
	case *syntax.Name:
		if fn, ok := c.pass.Info.Uses[x].(*types2.Func); ok {
			return fn, nil
		}
	case *syntax.SelectorExpr:
		if fn, ok := c.pass.Info.Uses[x.Sel].(*types2.Func); ok {
			return fn, x.X
		}
	case *syntax.IndexExpr:
		return c.callbackTarget(x.X)
	}
	return nil, nil
}

func (c *lockCheck) valueCall(arg syntax.Expr) *syntax.CallExpr {
	arg = syntax.Unparen(arg)
	if call, ok := c.valueCalls[arg]; ok {
		return call
	}
	var call *syntax.CallExpr
	if fn, _ := c.callbackTarget(arg); fn != nil {
		if _, ok := c.decls[origin(fn)]; ok {
			call = &syntax.CallExpr{Fun: arg}
			call.SetPos(arg.Pos())
		}
	}
	c.valueCalls[arg] = call
	return call
}

// callbackBody returns the body of a function literal or a lambda, or nil.
func callbackBody(e syntax.Expr) syntax.Node {
	switch f := syntax.Unparen(e).(type) {
	case *syntax.FuncLit:
		return f.Body
	case *syntax.LambdaExpr:
		return f.Body
	}
	return nil
}

// immediateLiteral returns e when it is a directly invoked function
// literal or lambda, or nil.
func immediateLiteral(e syntax.Expr) syntax.Node {
	switch f := syntax.Unparen(e).(type) {
	case *syntax.FuncLit:
		return f
	case *syntax.LambdaExpr:
		return f
	}
	return nil
}

// syncVisitor walks one synchronous flow: a declaration body while
// building summaries, or a WithLock callback body while answering a
// query. A go statement contributes only its receiver and argument
// subexpressions; the invoked call is asynchronous and never a seed or a
// call edge. A function literal is traversed when it is a WithLock
// callback, the directly invoked function of a synchronous call, or an
// argument of a chain method call, and skipped otherwise: a callee that
// is not a chain method may store the literal, run it later, or spawn it.
// A group child block is skipped like a go body.
type syncVisitor struct {
	c      *lockCheck
	file   *syntax.File
	funLit map[syntax.Node]bool
	active map[syntax.Node]bool
	onCall func(*syntax.CallExpr)
}

// Visit implements syntax.Visitor.
func (v *syncVisitor) Visit(n syntax.Node) syntax.Visitor {
	switch n := n.(type) {
	case nil:
		return nil
	case *syntax.CallStmt:
		if n.Tok == syntax.Go {
			if call := callOf(n.Call); call != nil {
				v.c.asyncCall[call] = true
				v.visitAsyncOperands(call)
			}
			return nil
		}
		return v
	case *syntax.FuncLit:
		if v.c.callbacks[n] || v.funLit[n] {
			return v
		}
		return nil
	case *syntax.LambdaExpr:
		if v.c.callbacks[n] || v.funLit[n] {
			return v
		}
		return nil
	case *syntax.GroupChild:
		// A group child body runs in its own goroutine; excluded from
		// the synchronous flow like a go body. The group context and
		// limit still walk through the default case.
		return nil
	case *syntax.CallExpr:
		if v.c.asyncCall[n] {
			v.visitAsyncOperands(n)
			return nil
		}
		if lit := immediateLiteral(n.Fun); lit != nil {
			v.funLit[lit] = true
		}
		for _, fn := range chainFuncs(v.c.pass, n) {
			v.funLit[fn] = true
		}
		v.walkChainValues(n)
		v.walkStoredCall(n)
		if v.onCall != nil {
			v.onCall(n)
		}
		return v
	default:
		return v
	}
}

// visitAsyncOperands walks the synchronous subexpressions of an
// asynchronous invocation: the function operand unless it is a literal
// whose body runs in the new goroutine, and every argument.
func (v *syncVisitor) visitAsyncOperands(call *syntax.CallExpr) {
	if immediateLiteral(call.Fun) == nil {
		syntax.Walk(call.Fun, v)
	}
	for _, a := range call.ArgList {
		syntax.Walk(a, v)
	}
}

func (v *syncVisitor) walkChainValues(call *syntax.CallExpr) {
	if !isChainCall(v.c.pass, call) {
		return
	}
	for _, arg := range call.ArgList {
		if name, ok := syntax.Unparen(arg).(*syntax.Name); ok {
			v.walkStoredName(name)
		}
		if vc := v.c.valueCall(arg); vc != nil && v.onCall != nil {
			v.onCall(vc)
		}
	}
}

func (v *syncVisitor) walkStoredCall(call *syntax.CallExpr) {
	if immediateLiteral(call.Fun) != nil {
		return
	}
	if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok {
		v.walkStoredName(name)
	}
}

func (v *syncVisitor) walkStoredName(name *syntax.Name) {
	obj, ok := v.c.pass.Info.Uses[name]
	if !ok {
		return
	}
	body, ok := v.c.stored[obj]
	if !ok {
		return
	}
	if v.active == nil {
		v.active = make(map[syntax.Node]bool)
	}
	if v.active[body] {
		return
	}
	v.active[body] = true
	switch b := body.(type) {
	case *syntax.FuncLit:
		syntax.Walk(b.Body, v)
	case *syntax.LambdaExpr:
		syntax.Walk(b.Body, v)
	}
}

// buildSummaries computes the lock summary of every same-package function
// as the least fixed point of the seed and pull rules over the finite
// target set of function-key pairs within the path bound.
func (c *lockCheck) buildSummaries() {
	limit := 0
	for _, lc := range c.calls {
		if len(lc.key.path) > limit {
			limit = len(lc.key.path)
		}
	}
	c.keyPathLimit = limit
	if len(c.calls) == 0 {
		return
	}
	for o := range c.decls {
		if _, ok := c.summaries[o]; !ok {
			c.summaries[o] = &lockSummary{keys: make(map[lockKey]bool)}
		}
	}
	for fn, decl := range c.decls {
		file := c.declFiles[decl]
		v := &syncVisitor{c: c, file: file, funLit: make(map[syntax.Node]bool)}
		caller := fn
		v.onCall = func(cs *syntax.CallExpr) {
			c.noteSummaryCall(caller, decl, cs, file)
		}
		syntax.Walk(decl.Body, v)
	}
	var queue []*types2.Func
	for o, s := range c.summaries {
		if len(s.keys) > 0 {
			s.queued = true
			queue = append(queue, o)
		}
	}
	for len(queue) > 0 {
		g := queue[0]
		queue = queue[1:]
		gs := c.summaries[g]
		gs.queued = false
		for _, e := range c.callers[g] {
			fs := c.summaries[e.caller]
			gd := c.decls[e.callee]
			if fs == nil || gd == nil {
				continue
			}
			grew := false
			for kg := range gs.keys {
				sk := c.substitute(kg, e.call, gd)
				if sk.root == nil || len(sk.path) > c.keyPathLimit {
					continue
				}
				if !fs.keys[sk] {
					fs.keys[sk] = true
					grew = true
				}
			}
			if grew && !fs.queued {
				fs.queued = true
				queue = append(queue, e.caller)
			}
		}
		for _, e := range c.cbCallers[g] {
			fs := c.summaries[e.caller]
			if fs == nil {
				continue
			}
			grew := false
			for kg := range gs.keys {
				kc := c.bindKey(kg, e.bind, e.decl)
				if kc.root == nil || len(kc.path) > c.keyPathLimit {
					continue
				}
				if !fs.keys[kc] {
					fs.keys[kc] = true
					grew = true
				}
			}
			if grew && !fs.queued {
				fs.queued = true
				queue = append(queue, e.caller)
			}
		}
	}
}

// noteSummaryCall records the synchronous call cs of caller: an indexed
// WithLock call seeds the caller's summary, and a call to a same-package
// function records a caller edge for the worklist.
func (c *lockCheck) noteSummaryCall(caller *types2.Func, decl *syntax.FuncDecl, cs *syntax.CallExpr, file *syntax.File) {
	if c.asyncCall[cs] {
		return
	}
	if c.fileOf[cs] == nil {
		c.fileOf[cs] = file
	}
	if lc, ok := c.calls[cs]; ok {
		if len(lc.key.path) <= c.keyPathLimit {
			if s := c.summaries[caller]; s != nil {
				s.keys[lc.key] = true
			}
		}
		if lc.cbTarget != nil && lc.cbDecl != nil {
			c.cbCallers[lc.cbTarget] = append(c.cbCallers[lc.cbTarget], lockCallbackEdge{
				caller: caller,
				bind:   lc.bind,
				decl:   lc.cbDecl,
			})
		}
	}
	if g := calleeFunc(c.pass, cs); g != nil {
		if o := origin(g); o != nil {
			if _, ok := c.decls[o]; ok {
				c.callers[o] = append(c.callers[o], lockEdge{caller: caller, call: cs, callee: o})
			}
		}
	}
}

// checkOuter reports the re-entrant calls inside the callback of call, when
// call is a WithLock call with a callback.
func (c *lockCheck) checkOuter(call *syntax.CallExpr) {
	lc, ok := c.calls[call]
	if !ok || lc.body == nil || c.asyncCall[call] {
		return
	}
	file := lc.cbFile
	if file == nil {
		file = c.file
	}
	v := &syncVisitor{c: c, file: file, funLit: make(map[syntax.Node]bool)}
	v.onCall = func(cs *syntax.CallExpr) {
		if c.fileOf[cs] == nil {
			c.fileOf[cs] = file
		}
		c.checkInner(cs, lc.key, lc.bind, lc.cbDecl)
	}
	syntax.Walk(lc.body, v)
}

// checkInner reports call when it re-enters the lock named key: it is a
// WithLock call on key, or it calls a same-package function whose summary
// reaches key.
func (c *lockCheck) checkInner(call *syntax.CallExpr, key lockKey, bind map[types2.Object]lockKey, cbDecl *syntax.FuncDecl) {
	if lc, ok := c.calls[call]; ok {
		if c.bindKey(lc.key, bind, cbDecl) == key {
			c.flag(call, "WithLock on "+syntax.String(lc.recv)+" inside a WithLock callback of the same receiver deadlocks: the mutex is already held")
			return
		}
		if lc.cbTarget != nil {
			if tsum := c.summaries[lc.cbTarget]; tsum != nil {
				for tk := range tsum.keys {
					ck := c.bindKey(tk, lc.bind, lc.cbDecl)
					if ck.root == nil {
						continue
					}
					if c.bindKey(ck, bind, cbDecl) == key {
						c.flag(call, "call to "+lc.cbTarget.Name()+" takes WithLock on the receiver of the enclosing WithLock callback and deadlocks: the mutex is already held")
						return
					}
				}
			}
		}
		return
	}
	fn := calleeFunc(c.pass, call)
	if fn == nil {
		return
	}
	o := origin(fn)
	decl, ok := c.decls[o]
	if !ok {
		return
	}
	sum := c.summaries[o]
	if sum == nil {
		return
	}
	for sk := range sum.keys {
		if c.bindKey(c.substitute(sk, call, decl), bind, cbDecl) == key {
			c.flag(call, "call to "+fn.Name()+" takes WithLock on the receiver of the enclosing WithLock callback and deadlocks: the mutex is already held")
			return
		}
	}
}

// bindKey maps k, a key in a callback body frame, into the frame of the
// enclosing WithLock call: a method-value receiver root takes the
// call-site key, any other root declared by a named callback takes no key,
// and all other roots pass through unchanged. A literal callback shares
// its enclosing frame, so with no callback declaration every key passes
// through and matches by object identity.
func (c *lockCheck) bindKey(k lockKey, bind map[types2.Object]lockKey, cbDecl *syntax.FuncDecl) lockKey {
	if k.root == nil {
		return k
	}
	if len(bind) > 0 {
		if outer, ok := bind[k.root]; ok {
			return lockKey{root: outer.root, path: outer.path + k.path}
		}
	}
	if cbDecl != nil && c.isDeclared(k.root, cbDecl) {
		return lockKey{}
	}
	return k
}

// substitute returns key, a receiver key in the body of decl, as seen from
// call: a key rooted at the receiver or a parameter of decl takes the key of
// the matching operand of call. A key rooted at a parameter whose type is
// not a pointer cannot map: passing the argument copies the value, so a
// lock reached through it holds its own mutex. A key that cannot be mapped
// has no root.
func (c *lockCheck) substitute(key lockKey, call *syntax.CallExpr, decl *syntax.FuncDecl) lockKey {
	operand := c.operandFor(key.root, call, decl)
	if operand == nil {
		if key.root != nil && c.isDeclared(key.root, decl) {
			return lockKey{}
		}
		return key
	}
	if c.isValueParam(key.root, decl) {
		return lockKey{}
	}
	outer := c.keyOf(operand)
	if outer.root == nil {
		return lockKey{}
	}
	return lockKey{root: outer.root, path: outer.path + key.path}
}

// isDeclared reports whether obj is declared inside decl: its receiver, a
// parameter, or a local variable.
func (c *lockCheck) isDeclared(obj types2.Object, decl *syntax.FuncDecl) bool {
	pos := obj.Pos()
	return pos.IsKnown() && within(pos, syntax.StartPos(decl), syntax.EndPos(decl))
}

// isValueParam reports whether obj is a parameter of decl, but not its
// receiver, whose type is not a pointer.
func (c *lockCheck) isValueParam(obj types2.Object, decl *syntax.FuncDecl) bool {
	v, ok := obj.(*types2.Var)
	if !ok || v.Type() == nil || decl.Recv != nil && c.definedBy(decl.Recv) == obj {
		return false
	}
	for _, p := range decl.Type.ParamList {
		if c.definedBy(p) != obj {
			continue
		}
		_, isPtr := v.Type().Underlying().(*types2.Pointer)
		return !isPtr
	}
	return false
}

// operandFor returns the operand of call that binds the receiver or the
// parameter of decl that root names, or nil when root is neither.
func (c *lockCheck) operandFor(root types2.Object, call *syntax.CallExpr, decl *syntax.FuncDecl) syntax.Expr {
	if decl.Recv != nil && c.definedBy(decl.Recv) == root {
		if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok {
			return sel.X
		}
		return nil
	}
	for i, p := range decl.Type.ParamList {
		if c.definedBy(p) == root && i < len(call.ArgList) {
			return call.ArgList[i]
		}
	}
	return nil
}

// definedBy returns the object that a named parameter declares, or nil.
func (c *lockCheck) definedBy(f *syntax.Field) types2.Object {
	if f.Name == nil {
		return nil
	}
	return c.pass.Info.Defs[f.Name]
}

// keyOf returns the key of receiver expression e, or a key with no root
// when e is not a variable, a field selection chain from one, or an
// address-of or dereference of those.
func (c *lockCheck) keyOf(e syntax.Expr) lockKey {
	path := ""
	for e != nil {
		if name, ok := e.(*syntax.Name); ok {
			root := c.variableOf(name)
			if root != nil {
				if a, ok := c.aliases[root]; ok {
					return lockKey{root: a.root, path: a.path + path}
				}
			}
			return lockKey{root: root, path: path}
		}
		var field string
		e, field = c.keyStep(e)
		if field != "" {
			path = "." + field + path
		}
	}
	return lockKey{}
}

// variableOf returns the variable that name uses, or nil.
func (c *lockCheck) variableOf(name *syntax.Name) types2.Object {
	if v, ok := c.pass.Info.Uses[name].(*types2.Var); ok {
		return v
	}
	return nil
}

// keyStep returns the operand of e and the field e selects from it. An
// expression that cannot continue a key returns a nil operand.
func (c *lockCheck) keyStep(e syntax.Expr) (syntax.Expr, string) {
	switch x := e.(type) {
	case *syntax.ParenExpr:
		return x.X, ""
	case *syntax.Operation:
		if x.Y == nil && (x.Op == syntax.And || x.Op == syntax.Mul) {
			return x.X, ""
		}
	case *syntax.SelectorExpr:
		if sel := c.pass.Info.Selections[x]; sel != nil && sel.Kind() == types2.FieldVal {
			return x.X, x.Sel.Value
		}
	case *syntax.IndexExpr:
		if idx, ok := lockIndexStep(c.pass.Info, &c.indexIDs, x); ok {
			return x.X, "[" + idx + "]"
		}
	}
	return nil, ""
}

// flag reports call once.
func (c *lockCheck) flag(call *syntax.CallExpr, msg string) {
	if c.reported[call] {
		return
	}
	c.reported[call] = true
	f := c.fileOf[call]
	if f == nil {
		f = c.file
	}
	c.pass.Report(Diagnostic{
		Pos:     syntax.StartPos(call),
		End:     c.pass.CallEnd(f, call),
		Message: msg,
	})
}
