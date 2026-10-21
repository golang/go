// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"slices"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers the M6-S7 chain calls (SPEC Sec 1.4, T-104.6).
// Consecutive filter, map, take, and drop calls fuse into one loop
// with the receiver bound once; terminals end the loop; sortBy, get,
// and last materialize first. Lambdas become per-element closures, so
// no parameter substitution exists. A `?` in an expression lambda
// becomes an explicit error branch, closing produced io.Closers in
// reverse order. Every loop is one parsed fragment, so native tree and
// eject text agree by construction.
//
// The pass runs after the statement lowering: generated code holds no
// TryExpr, option splices drain through textOf, and tuple and lambda
// passes drain its splices in turn.

// chainResult is one enclosing result for zero synthesis.
type chainResult struct {
	typ  types2.Type
	expr syntax.Expr
}

// chainHoist is one hoisted error-checked computation.
type chainHoist struct {
	stmts []syntax.Stmt
	text  string
}

// chainOp is one fused operation. savedBody captures the lambda body
// before nested-chain lowering swaps it for a generated fragment:
// later reads must see the original subtree, whose spans address
// source, composed with nested splices. argSpan is the exact source
// span of the sole argument and bodySpan that of an expression-lambda
// body: syntax end positions stop before closing delimiters, so the
// token scan of the call bounds both.
type chainOp struct {
	method    string
	arg       syntax.Expr
	argSpan   span
	bodySpan  span
	lam       *syntax.LambdaExpr
	savedBody syntax.Node
	isTry     bool
	isBlk     bool
}

// lowerChains rewrites every chain of one file.
func (fl *fileLower) lowerChains() {
	if fl.loweredChains == nil {
		fl.loweredChains = make(map[*syntax.CallExpr]bool)
	}
	fl.transparentWrappers()
	for _, decl := range fl.file.DeclList {
		switch decl := decl.(type) {
		case *syntax.FuncDecl:
			if decl.Body != nil {
				decl.Body.List = fl.lowerChainList(decl.Body.List, fl.chainFuncResults(decl.Type))
			}
		case *syntax.VarDecl:
			fl.lowerChainPkgDecl(decl)
		}
	}
	fl.flushTypeImports()
}

// transparentWrappers records the generated option-wrapper function
// literals: a `?`-chain inside one belongs to the enclosing function,
// so collection descends through them instead of treating them as
// error boundaries.
func (fl *fileLower) transparentWrappers() {
	if fl.transparent == nil {
		fl.transparent = make(map[*syntax.FuncLit]bool)
	}
	for out := range fl.wrapSpan {
		call, ok := out.(*syntax.CallExpr)
		if !ok {
			continue
		}
		if lit, ok := syntax.Unparen(call.Fun).(*syntax.FuncLit); ok {
			fl.transparent[lit] = true
		}
	}
}

// chainFuncResults reports the enclosing results of a function type.
// The result is non-nil even when empty; nil means package level.
func (fl *fileLower) chainFuncResults(ft *syntax.FuncType) []chainResult {
	out := []chainResult{}
	for _, field := range ft.ResultList {
		out = append(out, chainResult{typ: fl.pkg.info.TypeOf(field.Type), expr: field.Type})
	}
	return out
}

// lowerChainList lowers chains in list with enclosing results,
// splicing hoists in place.
func (fl *fileLower) lowerChainList(list []syntax.Stmt, enc []chainResult) []syntax.Stmt {
	var out []syntax.Stmt
	for _, s := range list {
		hoists := fl.lowerChainStmt(s, enc)
		out = append(out, hoists...)
		out = append(out, s)
		for _, child := range tupleChildLists(s) {
			*child = fl.lowerChainList(*child, enc)
		}
		syntax.Inspect(s, func(n syntax.Node) bool {
			if lit, ok := n.(*syntax.FuncLit); ok {
				if fl.transparent[lit] {
					return true
				}
				lit.Body.List = fl.lowerChainList(lit.Body.List, fl.chainFuncResults(lit.Type))
				return false
			}
			return true
		})
	}
	return out
}

// lowerChainPkgDecl lowers chains in a package-level declaration.
// Function literals lower with their own results; a `?`-chain at
// package level never passed checking.
func (fl *fileLower) lowerChainPkgDecl(decl *syntax.VarDecl) {
	if decl.Values == nil {
		return
	}
	for _, call := range fl.chainsInDecl(decl) {
		for _, h := range fl.lowerChainTopLevel(call, nil, 0) {
			_ = h
			base.Fatalf("logos: ?-chain outside function at %s", call.Pos().String())
		}
	}
}

// chainsInDecl collects the outermost chain calls of a declaration in
// inner-first order, processing nested function literals inline.
func (fl *fileLower) chainsInDecl(decl *syntax.VarDecl) []*syntax.CallExpr {
	var all []*syntax.CallExpr
	syntax.Inspect(decl, func(n syntax.Node) bool {
		if lit, ok := n.(*syntax.FuncLit); ok {
			lit.Body.List = fl.lowerChainList(lit.Body.List, fl.chainFuncResults(lit.Type))
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok && fl.isChainCall(call) {
			all = append(all, call)
		}
		return true
	})
	return fl.outermostChains(all)
}

// outermostChains filters calls to those contained in no other,
// sorted inner-first. Each span is scanned once, before any lowering
// replaces nodes under the calls.
func (fl *fileLower) outermostChains(all []*syntax.CallExpr) []*syntax.CallExpr {
	spans := make(map[*syntax.CallExpr]span, len(all))
	for _, c := range all {
		cs, ce := fl.chainSpan(c)
		spans[c] = span{cs, ce}
	}
	var outer []*syntax.CallExpr
	for _, c := range all {
		contained := false
		for _, o := range all {
			if o != c && spans[o].start <= spans[c].start && spans[c].end <= spans[o].end &&
				(spans[o].start < spans[c].start || spans[c].end < spans[o].end) {
				contained = true
				break
			}
		}
		if !contained {
			outer = append(outer, c)
		}
	}
	slices.SortStableFunc(outer, func(a, b *syntax.CallExpr) int {
		return spans[a].end - spans[a].start - (spans[b].end - spans[b].start)
	})
	return outer
}

// lowerChainStmt lowers the outermost chains of one statement,
// returning hoists to prepend.
func (fl *fileLower) lowerChainStmt(s syntax.Stmt, enc []chainResult) []syntax.Stmt {
	calls := fl.chainsInStmt(s)
	if len(calls) == 0 {
		return nil
	}
	stmtStart := fl.stmtStartOf(s)
	var hoists []syntax.Stmt
	for _, call := range calls {
		for _, h := range fl.lowerChainTopLevel(call, enc, stmtStart) {
			hoists = append(hoists, h.stmts...)
			fl.add(stmtStart, stmtStart, h.text+"; ")
		}
	}
	return hoists
}

// stmtStartOf reports the offset of the first token of s, where hoisted
// computations go. A call or expression statement sits at its call
// parenthesis, so its start comes from the expression.
func (fl *fileLower) stmtStartOf(s syntax.Stmt) int {
	switch s := s.(type) {
	case *syntax.CallStmt:
		if s.Call != nil {
			return fl.exprStart(s.Call)
		}
	case *syntax.ExprStmt:
		if s.X != nil {
			return fl.exprStart(s.X)
		}
	}
	return fl.off(syntax.StartPos(s))
}

// lowerChainTopLevel lowers one outermost chain, returning hoists.
func (fl *fileLower) lowerChainTopLevel(call *syntax.CallExpr, enc []chainResult, stmtStart int) []chainHoist {
	cg := &chainGen{fl: fl, pos: call.Pos(), prefix: chainPrefix(call.Pos()), enc: enc, stmtStart: stmtStart}
	cg.run(call)
	return cg.hoists
}

// chainPrefix builds the per-site temporary prefix.
func chainPrefix(pos syntax.Pos) string {
	return fmt.Sprintf("_logos_ch%d_%d", pos.Line(), pos.Col())
}

// chainsInStmt collects the outermost chain calls of s in inner-first
// order, skipping nested function literals other than transparent
// option wrappers, whose chains belong to the enclosing function.
func (fl *fileLower) chainsInStmt(s syntax.Stmt) []*syntax.CallExpr {
	var all []*syntax.CallExpr
	syntax.Inspect(s, func(n syntax.Node) bool {
		if lit, ok := n.(*syntax.FuncLit); ok && !fl.transparent[lit] {
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok && fl.isChainCall(call) {
			all = append(all, call)
		}
		return true
	})
	return fl.outermostChains(all)
}

// chainSpan reports the full source span of a chain call.
func (fl *fileLower) chainSpan(call *syntax.CallExpr) (int, int) {
	return fl.exprStart(call), fl.chainCallEnd(call)
}

// exprStart reports the offset of the first token of e. A slice
// expression sits at its `[`, which syntax.StartPos reports as the
// start of everything it holds, so the descent runs through the
// operands of slices, indexes, selectors, assertions, and calls to the
// leftmost one, and StartPos takes over there. The descent keeps
// parenthesized operands whole, where leftmostOf would unwrap them and
// start inside the parentheses.
func (fl *fileLower) exprStart(e syntax.Expr) int {
	for {
		switch t := e.(type) {
		case *syntax.SliceExpr:
			e = t.X
		case *syntax.IndexExpr:
			e = t.X
		case *syntax.SelectorExpr:
			e = t.X
		case *syntax.AssertExpr:
			e = t.X
		case *syntax.CallExpr:
			e = t.Fun
		default:
			return fl.off(syntax.StartPos(e))
		}
	}
}

// receiverSpan reports the exact source span of a chain call's
// receiver: from its first token to the end of its last token before
// the `.` that opens the selector. The selector sits at that dot, so
// the token scan skips the comments and line breaks of a leading-dot
// continuation, and the span ends on the receiver's last token whatever
// its syntax shape.
func (fl *fileLower) receiverSpan(call *syntax.CallExpr) span {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		base.Fatalf("logos: chain call at %s has no receiver selector", call.Pos().String())
	}
	dot := fl.off(sel.Pos())
	if dot < 0 || dot >= len(fl.src) || fl.src[dot] != '.' {
		base.Fatalf("logos: chain call at %s has no source receiver", call.Pos().String())
	}
	start := fl.exprStart(sel.X)
	end := start
	if start < dot {
		end = lastTokenEnd(fl.src, start, dot)
	}
	if end <= start {
		base.Fatalf("logos: cannot locate chain receiver at %s", call.Pos().String())
	}
	return span{start: start, end: end}
}

// chainCallEnd reports the offset just past a call's closing paren.
// The call's position is its opening parenthesis, whatever the function
// operand looks like, and the token scan matches it. A degenerate span
// marks a normalized fragment node with no source parentheses to scan;
// it keeps its end position.
func (fl *fileLower) chainCallEnd(e syntax.Expr) int {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return fl.off(syntax.EndPos(e))
	}
	start := fl.off(syntax.StartPos(e))
	funEnd := fl.off(syntax.EndPos(syntax.Unparen(call.Fun)))
	if start >= funEnd {
		return fl.off(syntax.EndPos(e))
	}
	lparen := fl.off(call.Pos())
	if lparen < 0 || lparen >= len(fl.src) || fl.src[lparen] != '(' {
		base.Fatalf("logos: call at %s has no source argument list", call.Pos().String())
	}
	split, ok := splitCall(fl.src, lparen)
	if !ok {
		base.Fatalf("logos: call at %s has no closing parenthesis", call.Pos().String())
	}
	return split.end
}

// isChainCall reports whether e is a chain method call.
func (fl *fileLower) isChainCall(e syntax.Expr) bool {
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return false
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil {
		return false
	}
	switch sel.Sel.Value {
	case "filter", "map", "find", "any", "all", "sortBy",
		"first", "last", "get", "take", "drop", "sum", "count":
		// Array `.sum()` is the M14-S1 reduction, never a chain: the
		// checker accepts chain calls on slices only.
		if t := fl.pkg.info.TypeOf(sel.X); t != nil {
			if _, ok := t.Underlying().(*types2.Array); ok {
				return false
			}
		}
		return true
	}
	return false
}

// chainMethodOf reports the chain method of a chain call.
func chainMethodOf(call *syntax.CallExpr) string {
	if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok && sel.Sel != nil {
		return sel.Sel.Value
	}
	return ""
}

// isFusibleMethod reports whether a chain method fuses into the run loop.
func isFusibleMethod(m string) bool {
	switch m {
	case "filter", "map", "take", "drop":
		return true
	}
	return false
}

// decomposeChain splits a chain into the fused run ops in call order,
// the run base, and the terminal or barrier op, which is the outermost
// call when it is not fusible. The carrier is the innermost call of the
// run: the one whose receiver is the base.
func (fl *fileLower) decomposeChain(call *syntax.CallExpr) (ops []chainOp, base syntax.Expr, term chainOp, carrier *syntax.CallExpr) {
	rest := syntax.Expr(call)
	carrier = call
	if m := chainMethodOf(call); !isFusibleMethod(m) {
		term = fl.chainOpOf(call, m)
		if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok {
			rest = sel.X
		}
	}
	cur := rest
	for {
		c, ok := syntax.Unparen(cur).(*syntax.CallExpr)
		if !ok || !fl.isChainCall(c) {
			return ops, cur, term, carrier
		}
		m := chainMethodOf(c)
		if !isFusibleMethod(m) {
			return ops, cur, term, carrier
		}
		ops = append(ops, fl.chainOpOf(c, m))
		carrier = c
		if sel, ok := syntax.Unparen(c.Fun).(*syntax.SelectorExpr); ok {
			cur = sel.X
		} else {
			return ops, cur, term, carrier
		}
	}
}

// soleArgSpan reports the exact source span of the only argument of a
// chain call. The call's position is its opening parenthesis; the token
// scan skips comments and closes on the matching parenthesis, so the
// span ends on the argument's last token whatever its syntax shape.
func (fl *fileLower) soleArgSpan(call *syntax.CallExpr) span {
	lparen := fl.off(call.Pos())
	if lparen < 0 || lparen >= len(fl.src) || fl.src[lparen] != '(' {
		base.Fatalf("logos: chain call at %s has no source argument list", call.Pos().String())
	}
	split, ok := splitCall(fl.src, lparen)
	if !ok || len(split.args) != 1 {
		base.Fatalf("logos: cannot locate chain argument at %s", call.Pos().String())
	}
	return trimSpan(fl.src, split.args[0])
}

// lambdaBodySpan reports the exact source span of an expression-lambda
// body: from the first token past the arrow to the end of the lambda
// argument. A comment before the body stays out, since a line comment
// there would end the `return` statement the closure wraps around it.
func (fl *fileLower) lambdaBodySpan(lam *syntax.LambdaExpr, arg span) span {
	start := arg.end
	scanSegment(fl.src, fl.off(lam.Arrow)+len("=>"), arg.end, func(t tok) bool {
		if t.comment || t.auto {
			return true
		}
		start = t.start
		return false
	})
	return span{start: start, end: arg.end}
}

// chainOpOf reads one operation's argument and lambda flags.
func (fl *fileLower) chainOpOf(cur *syntax.CallExpr, method string) chainOp {
	op := chainOp{method: method}
	if len(cur.ArgList) == 1 {
		op.arg = cur.ArgList[0]
		op.argSpan = fl.soleArgSpan(cur)
		if lam, ok := syntax.Unparen(cur.ArgList[0]).(*syntax.LambdaExpr); ok {
			op.lam = lam
			op.bodySpan = fl.lambdaBodySpan(lam, op.argSpan)
			if body, ok := lam.Body.(syntax.Expr); ok {
				if _, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
					op.isTry = true
				}
			}
			if _, ok := lam.Body.(*syntax.BlockStmt); ok {
				op.isBlk = true
			}
		}
	}
	return op
}

// chainGen carries one chain emission.
type chainGen struct {
	fl        *fileLower
	pos       syntax.Pos
	prefix    string
	enc       []chainResult
	stmtStart int
	whole     span
	kept      []span
	hoists    []chainHoist
	elemT     types2.Type
}

// run lowers a chain call, appending hoists for nested `?`-chains.
// Chains nested in the op arguments lower first through child
// generators, so their splices compose into the closure texts and
// their hoists precede this run's hoist. Each call lowers once; the
// file records finished calls so later traversals over replaced trees
// never reprocess them.
func (cg *chainGen) run(call *syntax.CallExpr) {
	if cg.fl.loweredChains[call] {
		return
	}
	cg.fl.loweredChains[call] = true
	ops, base, term, carrier := cg.fl.decomposeChain(call)
	if len(ops) == 0 && term.method == "" {
		return
	}
	slices.Reverse(ops)
	recv := cg.fl.receiverSpan(carrier)
	cg.whole.start, cg.whole.end = cg.fl.chainSpan(call)
	cg.kept = keptSpans(recv, ops, term)
	for i := range ops {
		if ops[i].lam != nil {
			// D4: lower logos constructs in the body before the
			// closure captures its text: the fragment re-parses as
			// Go, so no `$"..."`, `??`, or property read may remain.
			cg.fl.preLowerCaptureBody(ops[i].lam)
			ops[i].savedBody = ops[i].lam.Body
		}
		cg.lowerArgChains(ops[i].arg)
	}
	if term.lam != nil {
		cg.fl.preLowerCaptureBody(term.lam)
		term.savedBody = term.lam.Body
	}
	cg.lowerArgChains(term.arg)
	baseText := cg.sub(base, recv)
	elemT := cg.baseElem(base)
	resultT := cg.fl.pkg.info.TypeOf(call)
	isTry := cg.runIsTry(ops, term)
	cg.emit(call, ops, baseText, elemT, resultT, term, isTry)
}

// preLowerCaptureBody lowers logos constructs in a lambda body before a
// capture pass bakes its text (D4, D7, D8): a chain closure re-parses
// its body as Go, and lambda/eject text must agree, so no `$"..."`,
// top-level `??`, or property read may remain raw. Top-level `??` and
// `?.` already lowered in the options pass; here property reads lower
// first, so interpolation operand texts drain their needles, and then
// interpolations lower innermost first. Every splice lands strictly
// inside the lambda span and drains into the capture text. Replaced
// bodies keep their checked types in optBodyType for prodType. The
// later full passes find no remaining nodes (chain bodies detach; plain
// bodies were renamed past re-matching), so nothing lowers twice.
func (fl *fileLower) preLowerCaptureBody(lam *syntax.LambdaExpr) {
	if lam == nil {
		return
	}
	// D7: stash the source span before any replacement: synthetic
	// replacements carry no scannable parens, but the capture text must
	// cover the original extent. A wrapper top (`??`/`?.`) already
	// records its exact span; a source call keeps tupleCallSpan.
	if body, ok := lam.Body.(syntax.Expr); ok {
		if _, done := fl.optBodySpan[lam]; !done {
			espan := span{fl.off(syntax.StartPos(body)), fl.off(syntax.EndPos(body))}
			if ws, ok := fl.wrapSpan[syntax.Unparen(body)]; ok {
				espan = ws
			} else if call, ok := syntax.Unparen(body).(*syntax.CallExpr); ok {
				espan = fl.callBodySpan(call)
			}
			if fl.optBodySpan == nil {
				fl.optBodySpan = make(map[*syntax.LambdaExpr]span)
			}
			fl.optBodySpan[lam] = espan
		}
	}
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		block.List = fl.lowerPropBlock(block.List)
	} else if body, ok := lam.Body.(syntax.Expr); ok {
		if t, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
			if call, ok := syntax.Unparen(t.X).(*syntax.CallExpr); ok {
				c := &propCollect{}
				t.X = fl.propWalk(call, c)
				fl.emitPropReads(c)
			}
		} else {
			orig := fl.pkg.info.TypeOf(body)
			c := &propCollect{}
			newBody := fl.propWalk(body, c)
			fl.emitPropReads(c)
			lam.Body = newBody
			fl.stashBodyType(orig, body, newBody)
		}
	} else {
		return
	}
	fl.preLowerCaptureInterps(lam.Body)
}

// stashBodyType records the checked type of a capture body the
// pre-pass replaced: the replacement is synthetic and carries no
// recorded type, but chain prodType still needs the produced type.
func (fl *fileLower) stashBodyType(orig types2.Type, old, new syntax.Expr) {
	if new == old || orig == nil {
		return
	}
	if fl.optBodyType == nil {
		fl.optBodyType = make(map[syntax.Expr]types2.Type)
	}
	fl.optBodyType[new] = orig
}

// preLowerCaptureInterps lowers the interpolated strings under root
// innermost first, crediting the scoped strconv/unsafe import needs to
// the final flush. Nodes without recorded kinds were already lowered.
func (fl *fileLower) preLowerCaptureInterps(root syntax.Node) {
	if root == nil {
		return
	}
	var lits []*syntax.InterpExpr
	syntax.Inspect(root, func(n syntax.Node) bool {
		if e, ok := n.(*syntax.InterpExpr); ok {
			lits = append(lits, e)
		}
		return true
	})
	slices.SortFunc(lits, func(a, b *syntax.InterpExpr) int {
		return len(a.Raw) - len(b.Raw)
	})
	for _, e := range lits {
		kinds, ok := interpKinds[e]
		if !ok {
			continue
		}
		delete(interpKinds, e)
		need, _ := fl.lowerInterpExpr(e, kinds)
		fl.interpNeedConv = fl.interpNeedConv || need
		if len(e.Exprs) > 0 {
			fl.interpNeedUnsafe = true
		}
	}
}

// keptSpans lists the source spans of a run whose text the generated
// loop reuses: the receiver and the call arguments. Comments of the
// chain outside them sit on the links between receiver, dots, method
// names, and parentheses, which the replacement would drop.
func keptSpans(recv span, ops []chainOp, term chainOp) []span {
	kept := []span{recv, term.argSpan}
	for _, op := range ops {
		kept = append(kept, op.argSpan)
	}
	return kept
}

// baseElem reports the element type of a run base expression.
func (cg *chainGen) baseElem(base syntax.Expr) types2.Type {
	t := cg.fl.pkg.info.TypeOf(base)
	if s, ok := types2.Unalias(t).(*types2.Slice); ok {
		return s.Elem()
	}
	return nil
}

// runIsTry reports whether a run contains a `?` lambda.
func (cg *chainGen) runIsTry(ops []chainOp, term chainOp) bool {
	for _, op := range ops {
		if op.isTry {
			return true
		}
	}
	return term.isTry
}

// lowerArgChains lowers the chains nested in one op argument: lambda
// bodies and index expressions. Nested function literals stay for the
// statement pass, which lowers them with their own results. Nested
// splices stay recorded so the saved body spans compose them later.
func (cg *chainGen) lowerArgChains(arg syntax.Expr) {
	if arg == nil {
		return
	}
	for _, n := range cg.nestedChains(arg) {
		cg.lowerNested(n)
	}
}

// sub lowers nested chains in the run base e and returns the composed
// text of its receiver span. A chain call in base position, such as a
// mid-chain sortBy, lowers as its own run; its value feeds the outer
// loop.
func (cg *chainGen) sub(e syntax.Expr, recv span) string {
	var nested []*syntax.CallExpr
	if bc, ok := syntax.Unparen(e).(*syntax.CallExpr); ok && cg.fl.isChainCall(bc) {
		nested = []*syntax.CallExpr{bc}
	} else {
		nested = cg.nestedChains(e)
	}
	for _, n := range nested {
		cg.lowerNested(n)
	}
	return cg.fl.textOf(recv)
}

// lowerNested lowers one nested chain through a child generator with
// its own prefix, merging its hoists ahead of this run's. Its splices
// stay recorded for the enclosing saved spans to compose.
func (cg *chainGen) lowerNested(call *syntax.CallExpr) {
	sub := &chainGen{fl: cg.fl, pos: call.Pos(), prefix: chainPrefix(call.Pos()), enc: cg.enc, stmtStart: cg.stmtStart}
	sub.run(call)
	cg.hoists = append(cg.hoists, sub.hoists...)
}

// nestedChains collects chain calls under e, inner-first, excluding e
// itself and nested function literals, which the statement pass lowers
// with their own results.
func (cg *chainGen) nestedChains(e syntax.Expr) []*syntax.CallExpr {
	var all []*syntax.CallExpr
	syntax.Inspect(e, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok && call != e && cg.fl.isChainCall(call) {
			all = append(all, call)
		}
		return true
	})
	return cg.fl.outermostChains(all)
}

// enclosingZeros spells the zero values of the enclosing non-error
// results for hoisted error returns.
func (cg *chainGen) enclosingZeros() string {
	var out []string
	for i, r := range cg.enc {
		if i == len(cg.enc)-1 {
			break
		}
		out = append(out, renderExpr(cg.fl.lambdaZero(r.typ, cg.pos)))
	}
	return strings.Join(out, ", ")
}

// chainEmit holds one loop emission.
type chainEmit struct {
	cg       *chainGen
	call     *syntax.CallExpr
	ops      []chainOp
	term     chainOp
	baseText string
	elemT    types2.Type
	resultT  types2.Type
	isTry    bool
	closures []string
	outElem  string
}

// prepare builds closure texts and the materialized element spelling.
// Each closure takes the element type its stage receives: the receiver
// element type until a map, then the production of the latest map; the
// materialized elements and the terminal closure see the final one.
func (ce *chainEmit) prepare() {
	cg := ce.cg
	elem := cg.spellType(cg.elemT)
	for i, op := range ce.ops {
		if op.lam != nil {
			ce.closures = append(ce.closures, cg.closure(cg.prefix+fmt.Sprintf("_f%d", i), op, elem, ce.opResult(op)))
		}
		if op.method == "map" {
			elem = ce.opResult(op)
		}
	}
	if ce.term.lam != nil {
		ce.closures = append(ce.closures, cg.closure(cg.prefix+"_ft", ce.term, elem, ce.opResult(ce.term)))
	}
	ce.outElem = elem
}

// rsuf returns the nil-error suffix for hoisted runs.
func (ce *chainEmit) rsuf() string {
	if ce.isTry {
		return ", nil"
	}
	return ""
}

// opResult reports the produced type spelling of a lambda op:
// predicates yield bool, mappers and keys their inferred types.
func (ce *chainEmit) opResult(op chainOp) string {
	cg := ce.cg
	switch op.method {
	case "filter", "find", "any", "all", "count":
		return "bool"
	case "sortBy":
		t := cgKeyType(cg, op)
		if t == nil {
			t = cg.elemT
		}
		return cg.spellType(t)
	default:
		t := cg.prodType(op)
		if t == nil {
			t = cg.elemT
		}
		return cg.spellType(t)
	}
}

// prodType reports the produced type of a map or sortBy op.
func (cg *chainGen) prodType(op chainOp) types2.Type {
	if op.lam == nil {
		return nil
	}
	if block, ok := op.savedBody.(*syntax.BlockStmt); ok {
		return cg.blockProdType(block)
	}
	body, ok := op.savedBody.(syntax.Expr)
	if !ok {
		return nil
	}
	if t, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
		if tv := cg.fl.pkg.info.TypeOf(t); tv != nil {
			return tv
		}
		return nil
	}
	if tv := cg.fl.pkg.info.TypeOf(body); tv != nil {
		return tv
	}
	// D4: a replaced capture body (top-level `??`/`?.`, property
	// read, interpolation) is synthetic and carries no recorded type;
	// the pre-lower stashed the checked type under the replacement.
	if st, ok := cg.fl.optBodyType[body]; ok {
		return st
	}
	return nil
}

// blockProdType infers a block body's produced type from its returns.
func (cg *chainGen) blockProdType(block *syntax.BlockStmt) types2.Type {
	var first types2.Type
	for _, s := range block.List {
		ret, ok := s.(*syntax.ReturnStmt)
		if !ok {
			continue
		}
		for _, res := range syntax.UnpackListExpr(ret.Results) {
			t := cg.fl.pkg.info.TypeOf(res)
			if t == nil {
				// D4: a return lowered by the capture pre-pass
				// (interpolation, property read) is synthetic; the
				// pre-pass stashed the checked type under it.
				t, _ = cg.fl.optBodyType[res]
			}
			if t == nil {
				continue
			}
			if b, ok := t.(*types2.Basic); ok && b.Kind() == types2.UntypedNil {
				continue
			}
			if isUntypedBasic(t) {
				t = defaultBasicType(t)
			}
			if first == nil {
				first = t
			}
		}
	}
	return first
}

// closure builds one per-element closure for a lambda op whose
// parameter has type elem. Pure bodies return the value; `?` bodies
// return value and error explicitly.
func (cg *chainGen) closure(name string, op chainOp, elem, result string) string {
	fl := cg.fl
	pname := op.lam.Params[0].Value
	if op.isTry {
		body, ok := op.savedBody.(syntax.Expr)
		if !ok {
			return ""
		}
		tryBody, ok := syntax.Unparen(body).(*syntax.TryExpr)
		if !ok {
			return ""
		}
		inner, ok := syntax.Unparen(tryBody.X).(*syntax.CallExpr)
		if !ok {
			return ""
		}
		callText := fl.textOf(span{fl.exprStart(inner), fl.chainCallEnd(inner)})
		zero := cg.zeroText(cg.inferType(op, result))
		vn := cg.prefix + "_v"
		en := cg.prefix + "_e"
		return fmt.Sprintf("%s := func(%s %s) (%s, error) { %s, %s := %s; if %s != nil { return %s, %s }; return %s, nil }", name, pname, elem, result, vn, en, callText, en, zero, en, vn)
	}
	if block, ok := op.savedBody.(*syntax.BlockStmt); ok {
		text := fl.textOf(span{fl.off(syntax.StartPos(block)), fl.off(syntax.EndPos(block))})
		inner := strings.TrimPrefix(text, "{")
		return fmt.Sprintf("%s := func(%s %s) %s {%s }", name, pname, elem, result, inner)
	}
	if expr, ok := op.savedBody.(syntax.Expr); ok {
		// D4: a top-level `??`/`?.` wrapper records its head insert
		// exactly on the body start and its tail exactly on the body
		// end, which textOf keeps pending. Drain those boundary
		// inserts here; strictly interior wrappers already drained.
		bodyText := fl.drainBodyWrapper(expr, fl.textOf(op.bodySpan))
		return fmt.Sprintf("%s := func(%s %s) %s { return %s }", name, pname, elem, result, bodyText)
	}
	return ""
}

// inferType resolves the checked type behind a result spelling.
func (cg *chainGen) inferType(op chainOp, result string) types2.Type {
	if op.lam == nil {
		return cg.elemT
	}
	if result == "bool" {
		return types2.Typ[types2.Bool]
	}
	t := cg.prodType(op)
	if t == nil {
		t = cg.elemT
	}
	return t
}

// zeroText spells the zero value of a type for error branches.
func (cg *chainGen) zeroText(t types2.Type) string {
	if t == nil {
		return "nil"
	}
	return renderExpr(cg.fl.lambdaZero(t, cg.pos))
}

// cgKeyType reports a sortBy key type for closures.
func cgKeyType(cg *chainGen, op chainOp) types2.Type {
	if op.lam == nil {
		return nil
	}
	if block, ok := op.savedBody.(*syntax.BlockStmt); ok {
		return cg.blockProdType(block)
	}
	body, ok := op.savedBody.(syntax.Expr)
	if !ok {
		return nil
	}
	if t, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
		if tv := cg.fl.pkg.info.TypeOf(t); tv != nil {
			return tv
		}
		return nil
	}
	if t := cg.fl.pkg.info.TypeOf(body); t != nil {
		if isUntypedBasic(t) {
			return defaultBasicType(t)
		}
		return t
	}
	// D4: replaced capture body; see prodType.
	if st, ok := cg.fl.optBodyType[body]; ok {
		if isUntypedBasic(st) {
			return defaultBasicType(st)
		}
		return st
	}
	return nil
}

// spellType spells a type, defaulting untyped basics.
func (cg *chainGen) spellType(t types2.Type) string {
	if t == nil {
		return "any"
	}
	if isUntypedBasic(t) {
		t = defaultBasicType(t)
	}
	if spelled, ok := cg.fl.spellLowered(t); ok {
		return spelled
	}
	if spelled, ok := cg.fl.spellType(t); ok {
		return spelled
	}
	return "any"
}

// loop builds the fused loop body lines and the epilogue for a run.
// It returns the prelude lines (receiver, closures, counters), the
// per-element lines, and the epilogue lines.
func (ce *chainEmit) loop() (prelude, body, epilogue []string) {
	cg := ce.cg
	p := cg.prefix
	prelude = append(prelude, p+"_rcv := "+ce.baseText)
	prelude = append(prelude, ce.closures...)
	if ce.needOut() {
		prelude = append(prelude, "var "+p+"_out []"+ce.outElem)
	}
	if ce.term.method == "count" {
		prelude = append(prelude, p+"_c := 0")
	}
	for i, op := range ce.ops {
		switch op.method {
		case "take":
			prelude = append(prelude, fmt.Sprintf("%s_tn%d := %s", p, i, ce.argText(op)))
		case "drop":
			prelude = append(prelude, fmt.Sprintf("%s_d%d := %s", p, i, ce.argText(op)), fmt.Sprintf("%s_di%d := 0", p, i))
		}
	}
	cur := p + "_e"
	loopHead := fmt.Sprintf("for _, %[1]s_e := range %[1]s_rcv {", p)
	if !ce.needElem() {
		loopHead = fmt.Sprintf("for range %[1]s_rcv {", p)
	}
	body = append(body, loopHead)
	for i, op := range ce.ops {
		if op.method == "take" {
			body = append(body, fmt.Sprintf("if %[1]s_tc%[2]d >= %[1]s_tn%[2]d { break }", p, i))
		}
	}
	for i, op := range ce.ops {
		fname := fmt.Sprintf("%s_f%d", p, i)
		switch op.method {
		case "filter":
			body = append(body, ce.filterLines(p, fname, cur, i)...)
		case "take":
			body = append(body, fmt.Sprintf("%s_tc%d++", p, i))
			prelude = append(prelude, fmt.Sprintf("%s_tc%d := 0", p, i))
		case "drop":
			body = append(body, fmt.Sprintf("if %[1]s_di%[2]d < %[1]s_d%[2]d { %[1]s_di%[2]d++; continue }", p, i))
		case "map":
			mname := fmt.Sprintf("%s_m%d", p, i)
			body = append(body, ce.mapLines(p, fname, cur, mname, i)...)
			cur = mname
		}
	}
	body = append(body, ce.terminalLines(p, cur)...)
	body = append(body, "}")
	epilogue = ce.epilogue(p)
	return prelude, body, epilogue
}

// needOut reports whether the run materializes elements: slice and
// sum, sort, last, and get terminals always append; value terminals
// append only when a `?` rollback may need produced elements.
func (ce *chainEmit) needOut() bool {
	switch ce.term.method {
	case "find", "any", "all", "count", "first":
		return ce.isTry && ce.hasMap()
	}
	return true
}

// needElem reports whether the loop body references the element: only
// a bare count over a takedropped run leaves it unused.
func (ce *chainEmit) needElem() bool {
	if ce.term.method != "count" || ce.term.lam != nil {
		return true
	}
	for _, op := range ce.ops {
		if op.method == "filter" || op.method == "map" {
			return true
		}
	}
	return false
}

// argText returns the composed text of an index argument.
func (ce *chainEmit) argText(op chainOp) string {
	cg := ce.cg
	if op.arg == nil {
		return "0"
	}
	return cg.fl.textOf(op.argSpan)
}

// filterLines emits a filter check, pure or `?`.
func (ce *chainEmit) filterLines(p, fname, cur string, i int) []string {
	if ce.ops[i].isTry {
		ok := fmt.Sprintf("%s_ok%d", p, i)
		en := fmt.Sprintf("%s_e%d", p, i)
		return []string{fmt.Sprintf("%s, %s := %s(%s); if %s != nil { %s }; if !%s { continue }", ok, en, fname, cur, en, ce.failStage(p, en), ok)}
	}
	return []string{fmt.Sprintf("if !%s(%s) { continue }", fname, cur)}
}

// mapLines emits a map step, pure or `?` with rollback.
func (ce *chainEmit) mapLines(p, fname, cur, mname string, i int) []string {
	if ce.ops[i].isTry {
		en := fmt.Sprintf("%s_e%d", p, i)
		return []string{fmt.Sprintf("%s, %s := %s(%s); if %s != nil { %s }", mname, en, fname, cur, en, ce.failStage(p, en))}
	}
	return []string{fmt.Sprintf("%s := %s(%s)", mname, fname, cur)}
}

// failStage spells the error return for a failing stage, rolling back
// already produced elements first when the run maps: appended elements
// always passed every map stage, while a mapless run holds only
// receiver elements the caller owns.
func (ce *chainEmit) failStage(p, errVar string) string {
	if ce.hasMap() {
		return ce.failProduct(p, errVar)
	}
	return ce.failPristine(p, errVar)
}

// failProduct spells the rollback plus error return for a failing stage
// with produced elements: they close in reverse before returning errVar.
func (ce *chainEmit) failProduct(p, errVar string) string {
	return ce.failWith(p, errVar, true)
}

// failPristine spells the error return for a failing stage with nothing
// produced yet: no rollback, just the return through errVar.
func (ce *chainEmit) failPristine(p, errVar string) string {
	return ce.failWith(p, errVar, false)
}

func (ce *chainEmit) failWith(p, errVar string, rollbackBefore bool) string {
	if rollbackBefore {
		return ce.rollback(p) + "; " + ce.failReturn(p, errVar)
	}
	return ce.failReturn(p, errVar)
}

// hasMap reports whether the run maps: appended elements are produced
// values that rollback must close, while a mapless run holds only
// receiver elements the caller owns.
func (ce *chainEmit) hasMap() bool {
	for _, op := range ce.ops {
		if op.method == "map" {
			return true
		}
	}
	return false
}

// failReturn spells the error return for the run result through errVar,
// which the failing branch declares.
func (ce *chainEmit) failReturn(p, errVar string) string {
	switch ce.term.method {
	case "", "sortBy":
		return "return nil, " + errVar
	case "find", "first", "last", "get":
		return "return " + ce.noneText(p) + ", " + errVar
	case "any", "all":
		return "return false, " + errVar
	case "count", "sum":
		return "return 0, " + errVar
	}
	return "return nil, " + errVar
}

// rollback spells the reverse Closer rollback over produced elements.
func (ce *chainEmit) rollback(p string) string {
	return fmt.Sprintf("for %[1]s_i := len(%[1]s_out) - 1; %[1]s_i >= 0; %[1]s_i-- { if %[1]s_c, %[1]s_ok := any(%[1]s_out[%[1]s_i]).(interface{ Close() error }); %[1]s_ok { %[1]s_c.Close() } }", p)
}

// terminalLines emits the terminal step inside the loop.
func (ce *chainEmit) terminalLines(p, cur string) []string {
	t := ce.term
	appendCur := []string{fmt.Sprintf("%s_out = append(%s_out, %s)", p, p, cur)}
	if t.method == "" {
		return appendCur
	}
	keep := func(lines []string) []string {
		if ce.isTry && ce.hasMap() {
			return append(appendCur, lines...)
		}
		return lines
	}
	switch t.method {
	case "find":
		return keep(ce.findLines(p, cur))
	case "any", "all":
		return keep(ce.anyLines(p, cur))
	case "count":
		return keep(ce.countLines(p, cur))
	case "sum":
		return appendCur
	case "first":
		return keep(ce.firstLines(p, cur))
	case "sortBy", "last", "get":
		return appendCur
	}
	return appendCur
}

// findLines emits a find step.
func (ce *chainEmit) findLines(p, cur string) []string {
	fname := p + "_ft"
	if ce.term.isTry {
		en := p + "_fte"
		return []string{fmt.Sprintf("if %[1]s_fv, %[2]s := %[3]s(%[4]s); %[2]s != nil { %[5]s } else if %[1]s_fv { return %[6]s%[7]s }", p, en, fname, cur, ce.failStage(p, en), ce.someText(p, cur), ce.rsuf())}
	}
	return []string{fmt.Sprintf("if %s(%s) { return %s%s }", fname, cur, ce.someText(p, cur), ce.rsuf())}
}

// someText spells a present option value.
func (ce *chainEmit) someText(p, cur string) string {
	ce.cg.fl.needRT = true
	return fmt.Sprintf("_logos_rt.Option[%s]{Value: %s, Present: true}", ce.outElem, cur)
}

// noneText spells an absent option value.
func (ce *chainEmit) noneText(p string) string {
	ce.cg.fl.needRT = true
	return fmt.Sprintf("_logos_rt.Option[%s]{}", ce.outElem)
}

// anyLines emits an any/all step.
func (ce *chainEmit) anyLines(p, cur string) []string {
	fname := p + "_ft"
	if ce.term.method == "all" {
		if ce.term.isTry {
			en := p + "_fte"
			return []string{fmt.Sprintf("if %[1]s_fv, %[2]s := %[3]s(%[4]s); %[2]s != nil { %[5]s } else if !%[1]s_fv { return false%[6]s }", p, en, fname, cur, ce.failStage(p, en), ce.rsuf())}
		}
		return []string{fmt.Sprintf("if !%s(%s) { return false%s }", fname, cur, ce.rsuf())}
	}
	if ce.term.isTry {
		en := p + "_fte"
		return []string{fmt.Sprintf("if %[1]s_fv, %[2]s := %[3]s(%[4]s); %[2]s != nil { %[5]s } else if %[1]s_fv { return true%[6]s }", p, en, fname, cur, ce.failStage(p, en), ce.rsuf())}
	}
	return []string{fmt.Sprintf("if %s(%s) { return true%s }", fname, cur, ce.rsuf())}
}

// countLines emits a count step.
func (ce *chainEmit) countLines(p, cur string) []string {
	if ce.term.lam == nil {
		if ce.isTry && ce.hasMap() {
			return []string{fmt.Sprintf("%s_out = append(%s_out, %s)", p, p, cur), p + "_c++"}
		}
		return []string{p + "_c++"}
	}
	fname := p + "_ft"
	if ce.term.isTry {
		en := p + "_fte"
		return []string{fmt.Sprintf("if %[1]s_fv, %[2]s := %[3]s(%[4]s); %[2]s != nil { %[5]s } else if %[1]s_fv { %[1]s_c++ }", p, en, fname, cur, ce.failStage(p, en))}
	}
	return []string{fmt.Sprintf("if %s(%s) { %s_c++ }", fname, cur, p)}
}

// firstLines emits a first step.
func (ce *chainEmit) firstLines(p, cur string) []string {
	return []string{"return " + ce.someText(p, cur) + ce.rsuf()}
}

// epilogue emits the post-loop lines for terminals.
func (ce *chainEmit) epilogue(p string) []string {
	switch ce.term.method {
	case "":
		return []string{"return " + p + "_out" + ce.rsuf()}
	case "find", "any", "all", "count":
		return ce.epilogueValue(p)
	case "sum":
		return ce.epilogueSum(p)
	case "first":
		return []string{"return " + ce.noneText(p) + ce.rsuf()}
	case "last", "get":
		return ce.epilogueIndex(p)
	case "sortBy":
		return ce.epilogueSort(p)
	}
	return []string{"return " + p + "_out" + ce.rsuf()}
}

// epilogueValue emits defaults after exhaustive loops.
func (ce *chainEmit) epilogueValue(p string) []string {
	switch ce.term.method {
	case "find":
		return []string{"return " + ce.noneText(p) + ce.rsuf()}
	case "any":
		return []string{"return false" + ce.rsuf()}
	case "all":
		return []string{"return true" + ce.rsuf()}
	case "count":
		return []string{"return " + p + "_c" + ce.rsuf()}
	}
	return []string{"return " + p + "_out" + ce.rsuf()}
}

// epilogueSum emits pairwise summation over the materialized run. The
// helper recurses, so it declares with var before assigning: a name is
// not in scope in its own := initializer.
func (ce *chainEmit) epilogueSum(p string) []string {
	t := ce.outElem
	sn := p + "_sum"
	return []string{
		fmt.Sprintf("var %[1]s func([]%[2]s) %[2]s", sn, t),
		fmt.Sprintf("%[1]s = func(%[2]s_s []%[3]s) %[3]s { switch len(%[2]s_s) { case 0: return 0; case 1: return %[2]s_s[0]; default: %[2]s_m := len(%[2]s_s) / 2; return %[1]s(%[2]s_s[:%[2]s_m]) + %[1]s(%[2]s_s[%[2]s_m:]) } }", sn, p, t),
		fmt.Sprintf("return %s(%s_out)%s", sn, p, ce.rsuf()),
	}
}

// epilogueIndex emits last/get indexing.
func (ce *chainEmit) epilogueIndex(p string) []string {
	if ce.term.method == "last" {
		return []string{fmt.Sprintf("if len(%[1]s_out) == 0 { return %[2]s%[3]s }; return %[4]s%[3]s", p, ce.noneText(p), ce.rsuf(), ce.someText(p, p+"_out[len("+p+"_out)-1]"))}
	}
	idx := ce.argText(ce.term)
	return []string{
		fmt.Sprintf("%[1]s_i := %[2]s", p, idx),
		fmt.Sprintf("if %[1]s_i < 0 || %[1]s_i >= len(%[1]s_out) { return %[2]s%[3]s }; return %[4]s%[3]s", p, ce.noneText(p), ce.rsuf(), ce.someText(p, p+"_out["+p+"_i]")),
	}
}

// epilogueSort emits stable sorting over the materialized run, with
// precomputed keys when the key can fail. Comparison runs through the
// rt Compare shim, so NaN keys order before non-NaN per SPEC T-104.6,
// and sorting runs through the rt stable shim so lowered files import
// only logos/rt.
func (ce *chainEmit) epilogueSort(p string) []string {
	ce.cg.fl.needRT = true
	rt := rtAlias
	fname := p + "_ft"
	e := ce.outElem
	if !ce.term.isTry {
		return []string{
			fmt.Sprintf("%[1]s.SliceStableFunc(%[2]s_out, func(%[2]s_a, %[2]s_b %[3]s) int { return %[1]s.Compare(%[4]s(%[2]s_a), %[4]s(%[2]s_b)) })", rt, p, e, fname),
			fmt.Sprintf("return %s_out%s", p, ce.rsuf()),
		}
	}
	k := ce.opResult(ce.term)
	keyErr := p + "_ke"
	return []string{
		fmt.Sprintf("%[1]s_keys := make([]%[2]s, len(%[1]s_out))", p, k),
		fmt.Sprintf("for %[1]s_i := range %[1]s_out { %[1]s_kv, %[1]s_ke := %[3]s(%[1]s_out[%[1]s_i]); if %[1]s_ke != nil { %[2]s }; %[1]s_keys[%[1]s_i] = %[1]s_kv }", p, ce.failStage(p, keyErr), fname),
		fmt.Sprintf("%[1]s_idx := make([]int, len(%[1]s_out))", p),
		fmt.Sprintf("for %[1]s_i := range %[1]s_idx { %[1]s_idx[%[1]s_i] = %[1]s_i }", p),
		fmt.Sprintf("%[1]s.SliceStableFunc(%[1]s_idx, func(%[1]s_a, %[1]s_b int) int { return %[2]s.Compare(%[1]s_keys[%[1]s_a], %[1]s_keys[%[1]s_b]) })", p, rt),
		fmt.Sprintf("%[1]s_sorted := make([]%[2]s, len(%[1]s_out))", p, e),
		fmt.Sprintf("for %[1]s_i, %[1]s_v := range %[1]s_idx { %[1]s_sorted[%[1]s_i] = %[1]s_out[%[1]s_v] }", p),
		fmt.Sprintf("return %s_sorted%s", p, ce.rsuf()),
	}
}

// emit assembles one chain: closures, loop, replacement, and hoists.
func (cg *chainGen) emit(call *syntax.CallExpr, ops []chainOp, baseText string, elemT, resultT types2.Type, term chainOp, isTry bool) {
	fl := cg.fl
	cg.elemT = elemT
	ce := &chainEmit{cg: cg, call: call, ops: ops, term: term, baseText: baseText, elemT: elemT, resultT: resultT, isTry: isTry}
	ce.prepare()
	prelude, body, epilogue := ce.runLoop()
	var b strings.Builder
	for _, line := range prelude {
		b.WriteString(line + "; ")
	}
	for _, line := range body {
		b.WriteString(line + "; ")
	}
	for _, line := range epilogue {
		b.WriteString(line + "; ")
	}
	loopText := leftoverComments(fl, cg.whole, cg.kept) + strings.TrimSpace(b.String())
	if !isTry {
		text := "func() " + ce.resultSpelling() + " { " + loopText + " }()"
		native := parseChainExprFrag(fl, text, call.Pos())
		replaceTupleExpr(fl.file, call, native)
		fl.add(cg.whole.start, cg.whole.end, text)
		return
	}
	if cg.enc == nil {
		base.Fatalf("logos: ?-chain outside function at %s", call.Pos().String())
	}
	text := "func() (" + ce.resultSpelling() + ", error) { " + loopText + " }"
	temp := cg.prefix + "_t"
	errName := cg.prefix + "_err"
	ret := errName
	if zeros := cg.enclosingZeros(); zeros != "" {
		ret = zeros + ", " + errName
	}
	hoistText := temp + ", " + errName + " := " + text + "(); if " + errName + " != nil { return " + ret + " }"
	hoistNative := parseChainStmtFrag(fl, hoistText, call.Pos())
	cg.hoists = append(cg.hoists, chainHoist{stmts: hoistNative, text: hoistText})
	native := nameOf(call.Pos(), temp)
	replaceTupleExpr(fl.file, call, native)
	fl.add(cg.whole.start, cg.whole.end, temp)
}

// resultSpelling spells the chain result type.
func (ce *chainEmit) resultSpelling() string {
	cg := ce.cg
	t := ce.resultT
	if s, ok := types2.Unalias(t).(*types2.Slice); ok {
		_ = s
		return "[]" + ce.outElem
	}
	if _, ok := types2.Unalias(t).(*types2.Option); ok {
		cg.fl.needRT = true
		if spelled, ok := cg.fl.spellLowered(t); ok {
			return spelled
		}
	}
	return cg.spellType(t)
}

// parseChainExprFrag parses text as an expression and normalizes
// positions to pos.
func parseChainExprFrag(fl *fileLower, text string, pos syntax.Pos) syntax.Expr {
	f, err := syntax.Parse(syntax.NewFileBase("logos-chain"), strings.NewReader("package logoschain\nvar _ = "+text+"\n"), nil, nil, 0)
	if err != nil || len(f.DeclList) != 1 {
		base.Fatalf("logos: cannot parse chain fragment %q: %v", text, err)
	}
	vd, ok := f.DeclList[0].(*syntax.VarDecl)
	if !ok {
		base.Fatalf("logos: cannot parse chain fragment %q", text)
	}
	vals := syntax.UnpackListExpr(vd.Values)
	if len(vals) != 1 {
		base.Fatalf("logos: cannot parse chain fragment %q", text)
	}
	e := vals[0]
	normalizeChainPos(e, pos)
	return e
}

// normalizeChainPos sets every node under root to pos, including the
// raw delimiter positions that never reach dwarfgen as nodes. A single
// fragment coordinate poisons scope order, so blocks, switches,
// selects, clauses, and composite literals take pos on their closers.
func normalizeChainPos(root syntax.Node, pos syntax.Pos) {
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == nil {
			return true
		}
		n.SetPos(pos)
		switch t := n.(type) {
		case *syntax.BlockStmt:
			t.Rbrace = pos
		case *syntax.SwitchStmt:
			t.Rbrace = pos
		case *syntax.SelectStmt:
			t.Rbrace = pos
		case *syntax.CaseClause:
			t.Colon = pos
		case *syntax.CommClause:
			t.Colon = pos
		case *syntax.CompositeLit:
			t.Rbrace = pos
		}
		return true
	})
}

// parseChainStmtFrag parses text as statements and normalizes
// positions to pos.
func parseChainStmtFrag(fl *fileLower, text string, pos syntax.Pos) []syntax.Stmt {
	f, err := syntax.Parse(syntax.NewFileBase("logos-chain"), strings.NewReader("package logoschain\nfunc _() {\n"+text+"\n}\n"), nil, nil, 0)
	if err != nil || len(f.DeclList) != 1 {
		base.Fatalf("logos: cannot parse chain fragment %q: %v", text, err)
	}
	fd, ok := f.DeclList[0].(*syntax.FuncDecl)
	if !ok || fd.Body == nil {
		base.Fatalf("logos: cannot parse chain fragment %q", text)
	}
	out := fd.Body.List
	for _, s := range out {
		normalizeChainPos(s, pos)
	}
	return out
}

// isUntypedBasic reports whether t is an untyped basic type.
func isUntypedBasic(t types2.Type) bool {
	b, ok := t.(*types2.Basic)
	if !ok {
		return false
	}
	k := b.Kind()
	return k == types2.UntypedBool || k == types2.UntypedInt || k == types2.UntypedRune || k == types2.UntypedFloat || k == types2.UntypedComplex || k == types2.UntypedString
}

// defaultBasicType returns the default type of an untyped basic.
func defaultBasicType(t types2.Type) types2.Type {
	b := t.(*types2.Basic)
	switch b.Kind() {
	case types2.UntypedBool:
		return types2.Typ[types2.Bool]
	case types2.UntypedRune:
		return types2.Typ[types2.Rune]
	case types2.UntypedFloat:
		return types2.Typ[types2.Float64]
	case types2.UntypedComplex:
		return types2.Typ[types2.Complex128]
	case types2.UntypedString:
		return types2.Typ[types2.String]
	default:
		return types2.Typ[types2.Int]
	}
}
