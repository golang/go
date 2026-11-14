// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"fmt"
	"go/constant"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

var unboundedChannelOp = Rule{
	ID:      "correctness::unbounded_channel_op",
	Group:   Correctness,
	Default: Deny,
	Go:      true,
	Check:   checkUnboundedChannelOp,
}

func checkUnboundedChannelOp(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if n.Body != nil {
					newChanUnit(pass, f, n).run()
				}
				return false
			case *syntax.FuncLit:
				newChanUnit(pass, f, n).run()
				return false
			}
			return true
		})
	}
}

type chanOp struct {
	start syntax.Pos
	end   syntax.Pos
	ch    *types2.Var
	what  string
}

type chanUnit struct {
	pass     *Pass
	file     *syntax.File
	root     syntax.Node
	free     bool
	loops    int
	lits     int
	depth    int
	forStack []*syntax.ForStmt
	defs     map[*types2.Var]int
	caps     map[*types2.Var]int64
	sent     map[*types2.Var]int
	looped   map[*types2.Var]bool
	ops      []chanOp
	// timers holds the locals that an assignment initializes with
	// time.NewTimer or time.NewTicker. The constructor is known before the
	// walk; the rest is collected in source order while walking.
	timers map[*types2.Var]*timerUse
	reruns []span
}

// timerUse tracks one local variable of the unit that an assignment
// initializes with time.NewTimer or time.NewTicker. The constructor is
// known before the walk; the rest is collected in source order while
// walking: defAt is the function depth, defIn and reset the innermost for
// statement, deferred holds the depths of the defer statements that exempt
// its Stop calls, src is the defining variable, and taint marks a mention
// outside the four allowed forms.
type timerUse struct {
	ctor     string
	def      bool
	defAt    int
	defIn    *syntax.ForStmt
	recv     bool
	reset    *syntax.ForStmt
	resetAt  syntax.Pos
	deferred []int
	src      *types2.Var
	taint    bool
}

func newChanUnit(pass *Pass, f *syntax.File, root syntax.Node) *chanUnit {
	return &chanUnit{pass: pass, file: f, root: root}
}

func (u *chanUnit) run() {
	u.defs = make(map[*types2.Var]int)
	u.caps = make(map[*types2.Var]int64)
	u.timers = make(map[*types2.Var]*timerUse)
	u.reruns = backJumps(u.root)
	syntax.Inspect(u.root, u.collect)
	syntax.Inspect(u.root, u.taintAll)
	syntax.Inspect(u.root, u.visit)
	for _, op := range u.ops {
		if op.ch != nil && u.bounded(op.ch) {
			continue
		}
		u.pass.Report(Diagnostic{
			Pos:     op.start,
			End:     op.end,
			Message: fmt.Sprintf("%s may block forever: put it in a select with a <-ctx.Done() case, or make the channel locally with a constant capacity that covers every send", op.what),
		})
	}
}

func (u *chanUnit) bounded(ch *types2.Var) bool {
	k, ok := u.caps[ch]
	return ok && u.defs[ch] == 1 && !u.looped[ch] && int64(u.sent[ch]) <= k
}

// taintAll classifies every mention of a tracked timer and marks the ones
// outside the four allowed forms as tainted, before any select is judged.
// Allowed are the declaring statement, defer t.Stop(), a t.Reset standing
// alone as a statement at the top of a loop body, and the operand t.C of a
// receive or a range; every other mention, including a read or assignment
// of t.C, taints.
func (u *chanUnit) taintAll(n syntax.Node) bool {
	if body, ok := funcBody(n); ok {
		u.depth++
		syntax.Inspect(body, u.taintAll)
		u.depth--
		return false
	}
	switch n := n.(type) {
	case *syntax.Name:
		u.timerUse(n)
	case *syntax.SelectorExpr:
		if t := u.timerVar(n.X); t != nil {
			t.taint = true
		}
	case *syntax.Operation:
		if op := recvOf(n); op != nil {
			u.allowRecvOperand(op.X)
			return false
		}
	case *syntax.RangeClause:
		u.allowRecvOperand(n.X)
		if n.Lhs != nil {
			syntax.Inspect(n.Lhs, u.taintAll)
		}
		return false
	case *syntax.CallStmt:
		if n.Tok == syntax.Defer {
			u.deferMention(n)
			return false
		}
	case *syntax.ForStmt:
		u.forBody(n)
		return false
	}
	return true
}

func funcBody(n syntax.Node) (syntax.Node, bool) {
	switch n := n.(type) {
	case *syntax.FuncLit:
		return n.Body, true
	case *syntax.LambdaExpr:
		return n.Body, true
	case *syntax.GroupChild:
		return n.Body, true
	}
	return nil, false
}

// allowRecvOperand spares the C field of a tracked timer when x is the
// operand of a receive, and walks x as ordinary code otherwise.
func (u *chanUnit) allowRecvOperand(x syntax.Expr) {
	if sel, ok := syntax.Unparen(x).(*syntax.SelectorExpr); ok && sel.Sel.Value == "C" && u.timerVar(sel.X) != nil {
		return
	}
	syntax.Inspect(x, u.taintAll)
}

// deferMention spares the plain Stop call of a defer statement, records
// the depth of its function, and walks every other mention under the
// statement as ordinary code.
func (u *chanUnit) deferMention(n *syntax.CallStmt) {
	if call, ok := syntax.Unparen(n.Call).(*syntax.CallExpr); ok {
		t, plain := u.deferStop(call)
		if plain {
			t.deferred = append(t.deferred, u.depth)
		} else {
			syntax.Inspect(n.Call, u.taintAll)
		}
	} else {
		syntax.Inspect(n.Call, u.taintAll)
	}
	if n.DeferAt != nil {
		syntax.Inspect(n.DeferAt, u.taintAll)
	}
}

// deferStop reports the tracked timer of a plain t.Stop() call that names
// its defining variable, for the top call of a defer statement.
func (u *chanUnit) deferStop(call *syntax.CallExpr) (*timerUse, bool) {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Stop" {
		return nil, false
	}
	t := u.timerVar(sel.X)
	if t == nil {
		return nil, false
	}
	id, plain := syntax.Unparen(sel.X).(*syntax.Name)
	return t, plain && u.timerName(id) == t.src
}

// forBody walks a for statement: the header as ordinary code, a qualifying
// top-of-body Reset through its arguments only, and every other direct
// statement as ordinary code.
func (u *chanUnit) forBody(n *syntax.ForStmt) {
	if n.Init != nil {
		syntax.Inspect(n.Init, u.taintAll)
	}
	if n.Cond != nil {
		syntax.Inspect(n.Cond, u.taintAll)
	}
	if n.Post != nil {
		syntax.Inspect(n.Post, u.taintAll)
	}
	if n.Body == nil {
		return
	}
	for _, st := range n.Body.List {
		if v := u.topReset(st); v != nil && u.timers[v] != nil {
			if es, ok := st.(*syntax.ExprStmt); ok {
				if call, ok := syntax.Unparen(es.X).(*syntax.CallExpr); ok {
					for _, a := range call.ArgList {
						syntax.Inspect(a, u.taintAll)
					}
				}
			}
			continue
		}
		syntax.Inspect(st, u.taintAll)
	}
}

func (u *chanUnit) collect(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		if lhs, rhs, ok := assignPairs(n); ok {
			u.note(lhs, rhs)
		}
	case *syntax.VarDecl:
		if lhs, rhs, ok := declPairs(n); ok {
			u.note(lhs, rhs)
		}
	}
	return true
}

func (u *chanUnit) note(lhs, rhs []syntax.Expr) {
	for i, l := range lhs {
		id, ok := syntax.Unparen(l).(*syntax.Name)
		if !ok {
			continue
		}
		v := localVar(u.pass, id)
		if v == nil {
			continue
		}
		u.defs[v]++
		if len(rhs) != len(lhs) {
			continue
		}
		if k, ok := u.capacity(rhs[i]); ok {
			u.caps[v] = k
		}
		if ctor := timerCtor(u.pass, rhs[i]); ctor != "" {
			u.timers[v] = &timerUse{ctor: ctor, src: v}
		} else if aid, ok := syntax.Unparen(rhs[i]).(*syntax.Name); ok {
			if w := localVar(u.pass, aid); w != nil {
				if tw := u.timers[w]; tw != nil {
					u.timers[v] = tw
				}
			}
		}
	}
}

func (u *chanUnit) capacity(x syntax.Expr) (int64, bool) {
	call := callOf(x)
	if call == nil || !callsBuiltin(u.pass, call, "make") {
		return 0, false
	}
	if _, ok := underOrNil(TypeOf(call)).(*types2.Chan); !ok {
		return 0, false
	}
	if len(call.ArgList) < 2 {
		return 0, true
	}
	val := call.ArgList[1].GetTypeInfo().Value
	if val == nil {
		return -1, true
	}
	k, exact := constant.Int64Val(constant.ToInt(val))
	if !exact {
		return -1, true
	}
	return k, true
}

func (u *chanUnit) visit(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.ForStmt:
		u.loop(n)
		return false
	case *syntax.SelectStmt:
		u.selectStmt(n)
		return false
	case *syntax.GroupChild:
		u.child(n.Body)
		return false
	case *syntax.CallExpr:
		if stageOf(u.pass, n) != nil {
			u.segmentCall(n)
			return false
		}
		if body := callbackBody(n.Fun); body != nil {
			u.invoked(n, body)
			return false
		}
	case *syntax.FuncLit:
		u.nested(n.Body)
		return false
	case *syntax.LambdaExpr:
		u.nested(n.Body)
		return false
	case *syntax.SelectorExpr:
		u.timerDrain(n)
	case *syntax.AssignStmt:
		if lhs, _, ok := assignPairs(n); ok {
			u.timerDefs(lhs)
		}
	case *syntax.VarDecl:
		if lhs, _, ok := declPairs(n); ok {
			u.timerDefs(lhs)
		}
	case *syntax.SendStmt:
		u.send(n)
	case *syntax.Operation:
		u.operation(n)
	}
	return true
}

// timerDrain marks the tracked timer a receive from x, a C field, drains.
func (u *chanUnit) timerDrain(x syntax.Expr) {
	sel, ok := syntax.Unparen(x).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "C" {
		return
	}
	if t := u.timerVar(sel.X); t != nil {
		t.recv = true
	}
}

func (u *chanUnit) operation(n *syntax.Operation) {
	if op := recvOf(n); op != nil {
		u.recv(op)
	}
}

func (u *chanUnit) walk(n syntax.Node) {
	if n != nil {
		syntax.Inspect(n, u.visit)
	}
}

func (u *chanUnit) loop(f *syntax.ForStmt) {
	u.rangeOp(f.Init)
	u.walk(f.Init)
	u.loops++
	u.forStack = append(u.forStack, f)
	u.walk(f.Cond)
	u.walk(f.Post)
	u.bodyTop(f)
	u.forStack = u.forStack[:len(u.forStack)-1]
	u.loops--
}

func (u *chanUnit) rangeOp(st syntax.SimpleStmt) {
	rc, ok := st.(*syntax.RangeClause)
	if !ok {
		return
	}
	t := TypeOf(rc.X)
	if t == nil {
		return
	}
	if _, isChan := types2.CoreType(t).(*types2.Chan); !isChan {
		return
	}
	u.record(rc.Pos(), u.exprEnd(rc.X), u.chanVar(rc.X), "range over "+syntax.String(rc.X))
}

func (u *chanUnit) nested(body syntax.Node) {
	u.lits++
	u.once(body)
	u.lits--
}

func (u *chanUnit) invoked(call *syntax.CallExpr, body syntax.Node) {
	u.once(body)
	for _, arg := range call.ArgList {
		u.walk(arg)
	}
}

func (u *chanUnit) once(body syntax.Node) {
	outer := u.free
	u.free = false
	u.depth++
	u.walk(body)
	u.depth--
	u.free = outer
}

func (u *chanUnit) child(body syntax.Node) {
	outer := u.free
	u.free = true
	u.depth++
	u.walk(body)
	u.depth--
	u.free = outer
}

// bodyTop walks the direct statements of a for body in order. A Reset call
// that stands alone as a statement of the body rearms the timer for the
// next wait.
func (u *chanUnit) bodyTop(f *syntax.ForStmt) {
	if f.Body == nil {
		return
	}
	for _, st := range f.Body.List {
		if v := u.topReset(st); v != nil {
			if t := u.timers[v]; t != nil {
				t.reset = f
				t.resetAt = st.Pos()
				t.recv = false
			}
			if es, ok := st.(*syntax.ExprStmt); ok {
				if call, ok := syntax.Unparen(es.X).(*syntax.CallExpr); ok {
					for _, a := range call.ArgList {
						u.walk(a)
					}
				}
			}
			continue
		}
		u.walk(st)
	}
}

func (u *chanUnit) segmentCall(call *syntax.CallExpr) {
	u.walk(call.Fun)
	for _, arg := range call.ArgList {
		if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
			u.lits++
			u.child(lam.Body)
			u.lits--
			continue
		}
		u.walk(arg)
	}
}

// topReset returns the variable a loop body statement rearms: a bare t.Reset
// call standing alone as the statement, or nil.
func (u *chanUnit) topReset(st syntax.Stmt) *types2.Var {
	es, ok := st.(*syntax.ExprStmt)
	if !ok {
		return nil
	}
	call, ok := syntax.Unparen(es.X).(*syntax.CallExpr)
	if !ok {
		return nil
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Reset" {
		return nil
	}
	id, ok := syntax.Unparen(sel.X).(*syntax.Name)
	if !ok {
		return nil
	}
	return localVar(u.pass, id)
}

func (u *chanUnit) selectStmt(s *syntax.SelectStmt) {
	outer := u.free
	covered := outer || u.selectCovered(s)
	for _, cc := range s.Body {
		u.free = covered
		operands := u.comm(cc.Comm)
		u.free = outer
		for _, x := range operands {
			u.walk(x)
		}
		for _, st := range cc.Body {
			u.walk(st)
		}
	}
}

func (u *chanUnit) comm(st syntax.SimpleStmt) []syntax.Node {
	switch st := st.(type) {
	case *syntax.SendStmt:
		u.send(st)
		return []syntax.Node{st.Chan, st.Value}
	case *syntax.ExprStmt:
		if op := recvOf(st.X); op != nil {
			u.recv(op)
			return []syntax.Node{op.X}
		}
		return []syntax.Node{st.X}
	case *syntax.AssignStmt:
		if op := commRecv(st); op != nil {
			u.recv(op)
			return []syntax.Node{st.Lhs, op.X}
		}
		return []syntax.Node{st.Lhs, st.Rhs}
	}
	return nil
}

func (u *chanUnit) send(s *syntax.SendStmt) {
	ch := u.chanVar(s.Chan)
	if ch != nil {
		u.count(ch)
	}
	u.record(syntax.StartPos(s), u.exprEnd(s.Value), ch, "send on "+syntax.String(s.Chan))
}

func (u *chanUnit) count(ch *types2.Var) {
	if u.sent == nil {
		u.sent = make(map[*types2.Var]int)
		u.looped = make(map[*types2.Var]bool)
	}
	if u.loops > 0 || u.lits > 0 {
		u.looped[ch] = true
		return
	}
	u.sent[ch]++
}

func (u *chanUnit) recv(op *syntax.Operation) {
	if isDoneCall(op.X) {
		return
	}
	u.record(syntax.StartPos(op), u.exprEnd(op.X), u.chanVar(op.X), "receive from "+syntax.String(op.X))
}

func (u *chanUnit) record(start, end syntax.Pos, ch *types2.Var, what string) {
	if !u.free {
		u.ops = append(u.ops, chanOp{start: start, end: end, ch: ch, what: what})
	}
}

func (u *chanUnit) exprEnd(x syntax.Expr) syntax.Pos {
	mode := syntax.Mode(0)
	if u.pass.Kind == KindLgo {
		mode = syntax.LogosMode
	}
	switch x := x.(type) {
	case *syntax.CallExpr:
		return u.pass.CallEnd(u.file, x)
	case *syntax.IndexExpr, *syntax.SliceExpr:
		if end, ok := syntax.BracketEnd(u.pass.Source(u.file), x.Pos(), mode); ok {
			return end
		}
	case *syntax.ParenExpr:
		if end, ok := syntax.ParenEnd(u.pass.Source(u.file), x.Pos(), mode); ok {
			return end
		}
	case *syntax.AssertExpr:
		open := syntax.MakePos(x.Pos().Base(), x.Pos().Line(), x.Pos().Col()+1)
		if end, ok := syntax.ParenEnd(u.pass.Source(u.file), open, mode); ok {
			return end
		}
	case *syntax.Operation:
		if x.Y != nil {
			return u.exprEnd(x.Y)
		}
		return u.exprEnd(x.X)
	}
	return syntax.EndPos(x)
}

func (u *chanUnit) chanVar(x syntax.Expr) *types2.Var {
	id, ok := syntax.Unparen(x).(*syntax.Name)
	if !ok {
		return nil
	}
	return localVar(u.pass, id)
}

func recvOf(x syntax.Expr) *syntax.Operation {
	op, ok := syntax.Unparen(x).(*syntax.Operation)
	if !ok || op.Op != syntax.Recv || op.Y != nil {
		return nil
	}
	return op
}

func commRecv(a *syntax.AssignStmt) *syntax.Operation {
	if a.Rhs == nil {
		return nil
	}
	return recvOf(a.Rhs)
}

// selectCovered reports whether every communication of s is bounded: the
// select has a case receiving from x.Done(), a default case, or a case
// receiving from a timer. A default case never blocks and a timer case ends
// the wait, so the other cases are bounded too; the timer receive is itself
// a bound and is not reported either.
func (u *chanUnit) selectCovered(s *syntax.SelectStmt) bool {
	for _, cc := range s.Body {
		if cc.Comm == nil {
			return true
		}
		if op := commOperation(cc.Comm); op != nil && (isDoneCall(op.X) || u.isTimeoutRecv(s.Pos(), op.X)) {
			return true
		}
	}
	return false
}

// isTimeoutRecv reports whether x is a receive that bounds a select: a call
// of time.After, or the C field of a timer or ticker that the function
// creates with time.NewTimer or time.NewTicker and has not stopped.
func (u *chanUnit) isTimeoutRecv(at syntax.Pos, x syntax.Expr) bool {
	if call, ok := syntax.Unparen(x).(*syntax.CallExpr); ok {
		return isTimeAfter(u.pass, call)
	}
	sel, ok := syntax.Unparen(x).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "C" {
		return false
	}
	return u.localTimerBound(at, sel)
}

// localTimerBound reports whether sel ends the wait of the select at
// position at: the timer is a local of the function the walk is under,
// declared once and never assigned again, not stopped, and for a timer, not
// received from and not waiting in a for statement that neither declares
// nor resets it. A deferred Stop exempts only its own function.
func (u *chanUnit) localTimerBound(at syntax.Pos, sel *syntax.SelectorExpr) bool {
	v := u.timerName(sel.X)
	if v == nil {
		return false
	}
	t := u.timers[v]
	if t == nil || !t.def || t.defAt != u.depth || u.defs[v] != 1 || u.defs[t.src] != 1 || t.taint {
		return false
	}
	for _, at := range t.deferred {
		if at != u.depth {
			return false
		}
	}
	if t.ctor == "NewTicker" {
		return true
	}
	armed, ok := t.armedFor(u.loopAt())
	return ok && !t.recv && !u.rerunsWithout(at, armed)
}

func (t *timerUse) armedFor(loop *syntax.ForStmt) (syntax.Pos, bool) {
	if loop != nil && t.reset == loop {
		return t.resetAt, true
	}
	return t.src.Pos(), loop == nil || t.defIn == loop
}

type span struct{ from, to syntax.Pos }

func (s span) holds(pos syntax.Pos) bool {
	return s.from.Cmp(pos) <= 0 && pos.Cmp(s.to) <= 0
}

func backJumps(root syntax.Node) []span {
	var out []span
	syntax.Inspect(root, func(n syntax.Node) bool {
		jump, ok := n.(*syntax.BranchStmt)
		if !ok || jump.Tok != syntax.Goto {
			return true
		}
		if label, ok := jump.Target.(*syntax.LabeledStmt); ok && label.Pos().Cmp(jump.Pos()) < 0 {
			out = append(out, span{label.Pos(), jump.Pos()})
		}
		return true
	})
	return out
}

func (u *chanUnit) rerunsWithout(stmt, armed syntax.Pos) bool {
	for _, s := range u.reruns {
		if s.holds(stmt) && !s.holds(armed) {
			return true
		}
	}
	return false
}

// timerCtor returns NewTimer or NewTicker when x is a call of that function
// of package time, or "".
func timerCtor(pass *Pass, x syntax.Expr) string {
	call := callOf(x)
	if call == nil {
		return ""
	}
	fn := calleeFunc(pass, call)
	if fn == nil || (fn.Name() != "NewTimer" && fn.Name() != "NewTicker") {
		return ""
	}
	sig, ok := fn.Type().(*types2.Signature)
	if !ok || sig.Recv() != nil || fn.Pkg() == nil || fn.Pkg().Path() != "time" {
		return ""
	}
	return fn.Name()
}

// timerDefs records the declaration of each tracked timer in lhs. Defs
// holds a name only when the statement declares it, so a plain assignment,
// a parameter, or a variable of an enclosing function declares nothing.
func (u *chanUnit) timerDefs(lhs []syntax.Expr) {
	for _, l := range lhs {
		id, ok := syntax.Unparen(l).(*syntax.Name)
		if !ok {
			continue
		}
		v, ok := u.pass.Info.Defs[id].(*types2.Var)
		if !ok || localVar(u.pass, id) == nil {
			continue
		}
		if t := u.timers[v]; t != nil && !t.def {
			t.def = true
			t.defAt = u.depth
			t.defIn = u.loopAt()
		}
	}
}

// timerUse taints the tracked timer the name mentions.
func (u *chanUnit) timerUse(id *syntax.Name) {
	v, ok := u.pass.Info.Uses[id].(*types2.Var)
	if !ok {
		return
	}
	if t := u.timers[v]; t != nil {
		t.taint = true
	}
}

// timerName returns the variable that x names, through parentheses and
// pointer indirection, or nil.
func (u *chanUnit) timerName(x syntax.Expr) *types2.Var {
	for {
		x = syntax.Unparen(x)
		op, ok := x.(*syntax.Operation)
		if !ok || op.Y != nil || (op.Op != syntax.Mul && op.Op != syntax.And) {
			break
		}
		x = op.X
	}
	id, ok := x.(*syntax.Name)
	if !ok {
		return nil
	}
	return localVar(u.pass, id)
}

// timerVar returns the tracked timer whose receiver x names, or nil.
func (u *chanUnit) timerVar(x syntax.Expr) *timerUse {
	return u.timers[u.timerName(x)]
}

// loopAt returns the innermost for statement open at the current node, or
// nil.
func (u *chanUnit) loopAt() *syntax.ForStmt {
	if len(u.forStack) == 0 {
		return nil
	}
	return u.forStack[len(u.forStack)-1]
}

func commOperation(st syntax.SimpleStmt) *syntax.Operation {
	switch st := st.(type) {
	case *syntax.ExprStmt:
		return recvOf(st.X)
	case *syntax.AssignStmt:
		return commRecv(st)
	}
	return nil
}

func isDoneCall(x syntax.Expr) bool {
	call, ok := syntax.Unparen(x).(*syntax.CallExpr)
	if !ok || len(call.ArgList) != 0 {
		return false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel.Value != "Done" {
		return false
	}
	_, isChan := underOrNil(TypeOf(call)).(*types2.Chan)
	return isChan
}
