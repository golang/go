// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package rangefunc rewrites range-over-func to code that doesn't use range-over-funcs.
Rewriting the construct in the front end, before noder, means the functions generated during
the rewrite are available in a noder-generated representation for inlining by the back end.

# Theory of Operation

The basic idea is to rewrite

	for x := range f {
		...
	}

into

	f(func(x T) bool {
		...
	})

But it's not usually that easy.

# Range variables

For a range not using :=, the assigned variables cannot be function parameters
in the generated body function. Instead, we allocate fake parameters and
start the body with an assignment. For example:

	for expr1, expr2 = range f {
		...
	}

becomes

	f(func(#p1 T1, #p2 T2) bool {
		expr1, expr2 = #p1, #p2
		...
	})

(All the generated variables have a # at the start to signal that they
are internal variables when looking at the generated code in a
debugger. Because variables have all been resolved to the specific
objects they represent, there is no danger of using plain "p1" and
colliding with a Go variable named "p1"; the # is just nice to have,
not for correctness.)

It can also happen that there are fewer range variables than function
arguments, in which case we end up with something like

	f(func(x T1, _ T2) bool {
		...
	})

or

	f(func(#p1 T1, #p2 T2, _ T3) bool {
		expr1, expr2 = #p1, #p2
		...
	})

# Return

If the body contains a "break", that break turns into "return false",
to tell f to stop. And if the body contains a "continue", that turns
into "return true", to tell f to proceed with the next value.
Those are the easy cases.

If the body contains a return or a break/continue/goto L, then we need
to rewrite that into code that breaks out of the loop and then
triggers that control flow. In general we rewrite

	for x := range f {
		...
	}

into

	{
		var #next int
		f(func(x T1) bool {
			...
			return true
		})
		... check #next ...
	}

The variable #next is an integer code that says what to do when f
returns. Each difficult statement sets #next and then returns false to
stop f.

A plain "return" rewrites to {#next = -1; return false}.
The return false breaks the loop. Then when f returns, the "check
#next" section includes

	if #next == -1 { return }

which causes the return we want.

Return with arguments is more involved. We need somewhere to store the
arguments while we break out of f, so we add them to the var
declaration, like:

	{
		var (
			#next int
			#r1 type1
			#r2 type2
		)
		f(func(x T1) bool {
			...
			{
				// return a, b
				#r1, #r2 = a, b
				#next = -2
				return false
			}
			...
			return true
		})
		if #next == -2 { return #r1, #r2 }
	}

TODO: What about:

	func f() (x bool) {
		for range g(&x) {
			return true
		}
	}

	func g(p *bool) func(func() bool) {
		return func(yield func() bool) {
			yield()
			// Is *p true or false here?
		}
	}

With this rewrite the "return true" is not visible after yield returns,
but maybe it should be?

# Checking

To permit checking that an iterator is well-behaved -- that is, that
it does not call the loop body again after it has returned false or
after the entire loop has exited (it might retain a copy of the body
function, or pass it to another goroutine) -- each generated loop has
its own #exitK flag that is checked before each iteration, and set both
at any early exit and after the iteration completes.

For example:

	for x := range f {
		...
		if ... { break }
		...
	}

becomes

	{
		var #exit1 bool
		f(func(x T1) bool {
			if #exit1 { runtime.panicrangeexit() }
			...
			if ... { #exit1 = true ; return false }
			...
			return true
		})
		#exit1 = true
	}

# Nested Loops

So far we've only considered a single loop. If a function contains a
sequence of loops, each can be translated individually. But loops can
be nested. It would work to translate the innermost loop and then
translate the loop around it, and so on, except that there'd be a lot
of rewriting of rewritten code and the overall traversals could end up
taking time quadratic in the depth of the nesting. To avoid all that,
we use a single rewriting pass that handles a top-most range-over-func
loop and all the range-over-func loops it contains at the same time.

If we need to return from inside a doubly-nested loop, the rewrites
above stay the same, but the check after the inner loop only says

	if #next < 0 { return false }

to stop the outer loop so it can do the actual return. That is,

	for range f {
		for range g {
			...
			return a, b
			...
		}
	}

becomes

	{
		var (
			#next int
			#r1 type1
			#r2 type2
		)
		var #exit1 bool
		f(func() {
			if #exit1 { runtime.panicrangeexit() }
			var #exit2 bool
			g(func() {
				if #exit2 { runtime.panicrangeexit() }
				...
				{
					// return a, b
					#r1, #r2 = a, b
					#next = -2
					#exit1, #exit2 = true, true
					return false
				}
				...
				return true
			})
			#exit2 = true
			if #next < 0 {
				return false
			}
			return true
		})
		#exit1 = true
		if #next == -2 {
			return #r1, #r2
		}
	}

Note that the #next < 0 after the inner loop handles both kinds of
return with a single check.

# Labeled break/continue of range-over-func loops

For a labeled break or continue of an outer range-over-func, we
use positive #next values. Any such labeled break or continue
really means "do N breaks" or "do N breaks and 1 continue".
We encode that as perLoopStep*N or perLoopStep*N+1 respectively.

Loops that might need to propagate a labeled break or continue
add one or both of these to the #next checks:

	if #next >= 2 {
		#next -= 2
		return false
	}

	if #next == 1 {
		#next = 0
		return true
	}

For example

	F: for range f {
		for range g {
			for range h {
				...
				break F
				...
				...
				continue F
				...
			}
		}
		...
	}

becomes

	{
		var #next int
		var #exit1 bool
		f(func() {
			if #exit1 { runtime.panicrangeexit() }
			var #exit2 bool
			g(func() {
				if #exit2 { runtime.panicrangeexit() }
				var #exit3 bool
				h(func() {
					if #exit3 { runtime.panicrangeexit() }
					...
					{
						// break F
						#next = 4
						#exit1, #exit2, #exit3 = true, true, true
						return false
					}
					...
					{
						// continue F
						#next = 3
						#exit2, #exit3 = true, true
						return false
					}
					...
					return true
				})
				#exit3 = true
				if #next >= 2 {
					#next -= 2
					return false
				}
				return true
			})
			#exit2 = true
			if #next >= 2 {
				#next -= 2
				return false
			}
			if #next == 1 {
				#next = 0
				return true
			}
			...
			return true
		})
		#exit1 = true
	}

Note that the post-h checks only consider a break,
since no generated code tries to continue g.

# Gotos and other labeled break/continue

The final control flow translations are goto and break/continue of a
non-range-over-func statement. In both cases, we may need to break out
of one or more range-over-func loops before we can do the actual
control flow statement. Each such break/continue/goto L statement is
assigned a unique negative #next value (below -2, since -1 and -2 are
for the two kinds of return). Then the post-checks for a given loop
test for the specific codes that refer to labels directly targetable
from that block. Otherwise, the generic

	if #next < 0 { return false }

check handles stopping the next loop to get one step closer to the label.

For example

	Top: print("start\n")
	for range f {
		for range g {
			...
			for range h {
				...
				goto Top
				...
			}
		}
	}

becomes

	Top: print("start\n")
	{
		var #next int
		var #exit1 bool
		f(func() {
			if #exit1 { runtime.panicrangeexit() }
			var #exit2 bool
			g(func() {
				if #exit2 { runtime.panicrangeexit() }
				...
				var #exit3 bool
				h(func() {
				if #exit3 { runtime.panicrangeexit() }
					...
					{
						// goto Top
						#next = -3
						#exit1, #exit2, #exit3 = true, true, true
						return false
					}
					...
					return true
				})
				#exit3 = true
				if #next < 0 {
					return false
				}
				return true
			})
			#exit2 = true
			if #next < 0 {
				return false
			}
			return true
		})
		#exit1 = true
		if #next == -3 {
			#next = 0
			goto Top
		}
	}

Labeled break/continue to non-range-over-funcs are handled the same
way as goto.

# Defers

The last wrinkle is handling defer statements. If we have

	for range f {
		defer print("A")
	}

we cannot rewrite that into

	f(func() {
		defer print("A")
	})

because the deferred code will run at the end of the iteration, not
the end of the containing function. To fix that, the runtime provides
a special hook that lets us obtain a defer "token" representing the
outer function and then use it in a later defer to attach the deferred
code to that outer function.

Normally,

	defer print("A")

compiles to

	runtime.deferproc(func() { print("A") })

This changes in a range-over-func. For example:

	for range f {
		defer print("A")
	}

compiles to

	var #defers = runtime.deferrangefunc()
	f(func() {
		runtime.deferprocat(func() { print("A") }, #defers)
	})

For this rewriting phase, we insert the explicit initialization of
#defers and then attach the #defers variable to the CallStmt
representing the defer. That variable will be propagated to the
backend and will cause the backend to compile the defer using
deferprocat instead of an ordinary deferproc.

TODO: Could call runtime.deferrangefuncend after f.
*/
package rangefunc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
	"fmt"
	"go/constant"
	"os"
)

// nopos is the zero syntax.Pos.
var nopos syntax.Pos

// A rewriter implements rewriting the range-over-funcs in a given function.
type rewriter struct {
	pkg   *types2.Package
	info  *types2.Info
	outer *syntax.FuncType
	body  *syntax.BlockStmt

	// References to important types and values.
	any   types2.Object
	bool  types2.Object
	int   types2.Object
	true  types2.Object
	false types2.Object

	// Branch numbering, computed as needed.
	branchNext map[branch]int             // branch -> #next value
	labelLoop  map[string]*syntax.ForStmt // label -> innermost rangefunc loop it is declared inside (nil for no loop)

	// Stack of nodes being visited.
	stack    []syntax.Node // all nodes
	forStack []*forLoop    // range-over-func loops

	rewritten map[*syntax.ForStmt]syntax.Stmt

	// Declared variables in generated code for outermost loop.
	declStmt     *syntax.DeclStmt
	nextVar      types2.Object
	retVars      []types2.Object
	defers       types2.Object
	exitVarCount int // exitvars are referenced from their respective loops
}

// A branch is a single labeled branch.
type branch struct {
	tok   syntax.Token
	label string
}

// A forLoop describes a single range-over-func loop being processed.
type forLoop struct {
	nfor         *syntax.ForStmt // actual syntax
	exitFlag     *types2.Var     // #exit variable for this loop
	exitFlagDecl *syntax.VarDecl

	checkRet      bool     // add check for "return" after loop
	checkRetArgs  bool     // add check for "return args" after loop
	checkBreak    bool     // add check for "break" after loop
	checkContinue bool     // add check for "continue" after loop
	checkBranch   []branch // add check for labeled branch after loop
}

// Rewrite rewrites all the range-over-funcs in the files.
func Rewrite(pkg *types2.Package, info *types2.Info, files []*syntax.File) {
	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				rewriteFunc(pkg, info, n.Type, n.Body)
				return false
			case *syntax.FuncLit:
				rewriteFunc(pkg, info, n.Type, n.Body)
				return false
			}
			return true
		})
	}
}

// rewriteFunc rewrites all the range-over-funcs in a single function (a top-level func or a func literal).
// The typ and body are the function's type and body.
func rewriteFunc(pkg *types2.Package, info *types2.Info, typ *syntax.FuncType, body *syntax.BlockStmt) {
	if body == nil {
		return
	}
	r := &rewriter{
		pkg:   pkg,
		info:  info,
		outer: typ,
		body:  body,
	}
	syntax.Inspect(body, r.inspect)
	if (base.Flag.W != 0) && r.forStack != nil {
		syntax.Fdump(os.Stderr, body)
	}
}

// checkFuncMisuse reports whether to check for misuse of iterator callbacks functions.
func (r *rewriter) checkFuncMisuse() bool {
	return base.Debug.RangeFuncCheck != 0
}

// inspect is a callback for syntax.Inspect that drives the actual rewriting.
// If it sees a func literal, it kicks off a separate rewrite for that literal.
// Otherwise, it maintains a stack of range-over-func loops and
// converts each in turn.
func (r *rewriter) inspect(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.FuncLit:
		rewriteFunc(r.pkg, r.info, n.Type, n.Body)
		return false

	default:
		// Push n onto stack.
		r.stack = append(r.stack, n)
		if nfor, ok := forRangeFunc(n); ok {
			loop := &forLoop{nfor: nfor}
			r.forStack = append(r.forStack, loop)
			r.startLoop(loop)
		}

	case nil:
		// n == nil signals that we are done visiting
		// the top-of-stack node's children. Find it.
		n = r.stack[len(r.stack)-1]

		// If we are inside a range-over-func,
		// take this moment to replace any break/continue/goto/return
		// statements directly contained in this node.
		// Also replace any converted for statements
		// with the rewritten block.
		switch n := n.(type) {
		case *syntax.BlockStmt:
			for i, s := range n.List {
				n.List[i] = r.editStmt(s)
			}
		case *syntax.CaseClause:
			for i, s := range n.Body {
				n.Body[i] = r.editStmt(s)
			}
		case *syntax.CommClause:
			for i, s := range n.Body {
				n.Body[i] = r.editStmt(s)
			}
		case *syntax.LabeledStmt:
			n.Stmt = r.editStmt(n.Stmt)
		}

		// Pop n.
		if len(r.forStack) > 0 && r.stack[len(r.stack)-1] == r.forStack[len(r.forStack)-1].nfor {
			r.endLoop(r.forStack[len(r.forStack)-1])
			r.forStack = r.forStack[:len(r.forStack)-1]
		}
		r.stack = r.stack[:len(r.stack)-1]
	}
	return true
}

// startLoop sets up for converting a range-over-func loop.
func (r *rewriter) startLoop(loop *forLoop) {
	// For first loop in function, allocate syntax for any, bool, int, true, and false.
	if r.any == nil {
		r.any = types2.Universe.Lookup("any")
		r.bool = types2.Universe.Lookup("bool")
		r.int = types2.Universe.Lookup("int")
		r.true = types2.Universe.Lookup("true")
		r.false = types2.Universe.Lookup("false")
		r.rewritten = make(map[*syntax.ForStmt]syntax.Stmt)
	}
	if r.checkFuncMisuse() {
		// declare the exit flag for this loop's body
		loop.exitFlag, loop.exitFlagDecl = r.exitVar(loop.nfor.Pos())
	}
}

// editStmt returns the replacement for the statement x,
// or x itself if it should be left alone.
// This includes the for loops we are converting,
// as left in x.rewritten by r.endLoop.
func (r *rewriter) editStmt(x syntax.Stmt) syntax.Stmt {
	if x, ok := x.(*syntax.ForStmt); ok {
		if s := r.rewritten[x]; s != nil {
			return s
		}
	}

	if len(r.forStack) > 0 {
		switch x := x.(type) {
		case *syntax.BranchStmt:
			return r.editBranch(x)
		case *syntax.CallStmt:
			if x.Tok == syntax.Defer {
				return r.editDefer(x)
			}
		case *syntax.ReturnStmt:
			return r.editReturn(x)
		}
	}

	return x
}

// editDefer returns the replacement for the defer statement x.
// See the "Defers" section in the package doc comment above for more context.
func (r *rewriter) editDefer(x *syntax.CallStmt) syntax.Stmt {
	if r.defers == nil {
		// Declare and initialize the #defers token.
		init := &syntax.CallExpr{
			Fun: runtimeSym(r.info, "deferrangefunc"),
		}
		tv := syntax.TypeAndValue{Type: r.any.Type()}
		tv.SetIsValue()
		init.SetTypeInfo(tv)
		r.defers = r.declVar("#defers", r.any.Type(), init)
	}

	// Attach the token as an "extra" argument to the defer.
	x.DeferAt = r.useVar(r.defers)
	setPos(x.DeferAt, x.Pos())
	return x
}

func (r *rewriter) exitVar(pos syntax.Pos) (*types2.Var, *syntax.VarDecl) {
	r.exitVarCount++

	name := fmt.Sprintf("#exit%d", r.exitVarCount)
	typ := r.bool.Type()
	obj := types2.NewVar(pos, r.pkg, name, typ)
	n := syntax.NewName(pos, name)
	setValueType(n, typ)
	r.info.Defs[n] = obj

	return obj, &syntax.VarDecl{NameList: []*syntax.Name{n}}
}

// editReturn returns the replacement for the return statement x.
// See the "Return" section in the package doc comment above for more context.
func (r *rewriter) editReturn(x *syntax.ReturnStmt) syntax.Stmt {
	// #next = -1 is return with no arguments; -2 is return with arguments.
	var next int
	if x.Results == nil {
		next = -1
		r.forStack[0].checkRet = true
	} else {
		next = -2
		r.forStack[0].checkRetArgs = true
	}

	// Tell the loops along the way to check for a return.
	for _, loop := range r.forStack[1:] {
		loop.checkRet = true
	}

	// Assign results, set #next, and return false.
	bl := &syntax.BlockStmt{}
	if x.Results != nil {
		if r.retVars == nil {
			for i, a := range r.outer.ResultList {
				obj := r.declVar(fmt.Sprintf("#r%d", i+1), a.Type.GetTypeInfo().Type, nil)
				r.retVars = append(r.retVars, obj)
			}
		}
		bl.List = append(bl.List, &syntax.AssignStmt{Lhs: r.useList(r.retVars), Rhs: x.Results})
	}
	bl.List = append(bl.List, &syntax.AssignStmt{Lhs: r.next(), Rhs: r.intConst(next)})
	if r.checkFuncMisuse() {
		// mark all enclosing loop bodies as exited
		for i := 0; i < len(r.forStack); i++ {
			bl.List = append(bl.List, r.setExitedAt(i))
		}
	}
	bl.List = append(bl.List, &syntax.ReturnStmt{Results: r.useVar(r.false)})
	setPos(bl, x.Pos())
	return bl
}

// perLoopStep is part of the encoding of loop-spanning control flow
// for function range iterators.  Each multiple of two encodes a "return false"
// passing control to an enclosing iterator; a terminal value of 1 encodes
// "return true" (i.e., local continue) from the body function, and a terminal
// value of 0 encodes executing the remainder of the body function.
const perLoopStep = 2

// editBranch returns the replacement for the branch statement x,
// or x itself if it should be left alone.
// See the package doc comment above for more context.
func (r *rewriter) editBranch(x *syntax.BranchStmt) syntax.Stmt {
	if x.Tok == syntax.Fallthrough {
		// Fallthrough is unaffected by the rewrite.
		return x
	}

	// Find target of break/continue/goto in r.forStack.
	// (The target may not be in r.forStack at all.)
	targ := x.Target
	i := len(r.forStack) - 1
	if x.Label == nil && r.forStack[i].nfor != targ {
		// Unlabeled break or continue that's not nfor must be inside nfor. Leave alone.
		return x
	}
	for i >= 0 && r.forStack[i].nfor != targ {
		i--
	}
	// exitFrom is the index of the loop interior to the target of the control flow,
	// if such a loop exists (it does not if i == len(r.forStack) - 1)
	exitFrom := i + 1

	// Compute the value to assign to #next and the specific return to use.
	var next int
	var ret *syntax.ReturnStmt
	if x.Tok == syntax.Goto || i < 0 {
		// goto Label
		// or break/continue of labeled non-range-over-func loop.
		// We may be able to leave it alone, or we may have to break
		// out of one or more nested loops and then use #next to signal
		// to complete the break/continue/goto.
		// Figure out which range-over-func loop contains the label.
		r.computeBranchNext()
		nfor := r.forStack[len(r.forStack)-1].nfor
		label := x.Label.Value
		targ := r.labelLoop[label]
		if nfor == targ {
			// Label is in the innermost range-over-func loop; use it directly.
			return x
		}

		// Set #next to the code meaning break/continue/goto label.
		next = r.branchNext[branch{x.Tok, label}]

		// Break out of nested loops up to targ.
		i := len(r.forStack) - 1
		for i >= 0 && r.forStack[i].nfor != targ {
			i--
		}
		exitFrom = i + 1

		// Mark loop we exit to get to targ to check for that branch.
		// When i==-1 that's the outermost func body
		top := r.forStack[i+1]
		top.checkBranch = append(top.checkBranch, branch{x.Tok, label})

		// Mark loops along the way to check for a plain return, so they break.
		for j := i + 2; j < len(r.forStack); j++ {
			r.forStack[j].checkRet = true
		}

		// In the innermost loop, use a plain "return false".
		ret = &syntax.ReturnStmt{Results: r.useVar(r.false)}
	} else {
		// break/continue of labeled range-over-func loop.
		depth := len(r.forStack) - 1 - i

		// For continue of innermost loop, use "return true".
		// Otherwise we are breaking the innermost loop, so "return false".

		if depth == 0 && x.Tok == syntax.Continue {
			ret = &syntax.ReturnStmt{Results: r.useVar(r.true)}
			setPos(ret, x.Pos())
			return ret
		}
		ret = &syntax.ReturnStmt{Results: r.useVar(r.false)}

		// If this is a simple break, mark this loop as exited and return false.
		// No adjustments to #next.
		if depth == 0 {
			var stmts []syntax.Stmt
			if r.checkFuncMisuse() {
				stmts = []syntax.Stmt{r.setExited(), ret}
			} else {
				stmts = []syntax.Stmt{ret}
			}
			bl := &syntax.BlockStmt{
				List: stmts,
			}
			setPos(bl, x.Pos())
			return bl
		}

		// The loop inside the one we are break/continue-ing
		// needs to make that happen when we break out of it.
		if x.Tok == syntax.Continue {
			r.forStack[exitFrom].checkContinue = true
		} else {
			exitFrom = i
			r.forStack[exitFrom].checkBreak = true
		}

		// The loops along the way just need to break.
		for j := exitFrom + 1; j < len(r.forStack); j++ {
			r.forStack[j].checkBreak = true
		}

		// Set next to break the appropriate number of times;
		// the final time may be a continue, not a break.
		next = perLoopStep * depth
		if x.Tok == syntax.Continue {
			next--
		}
	}

	// Assign #next = next and do the return.
	as := &syntax.AssignStmt{Lhs: r.next(), Rhs: r.intConst(next)}
	bl := &syntax.BlockStmt{
		List: []syntax.Stmt{as},
	}

	if r.checkFuncMisuse() {
		// Set #exitK for this loop and those exited by the control flow.
		for i := exitFrom; i < len(r.forStack); i++ {
			bl.List = append(bl.List, r.setExitedAt(i))
		}
	}

	bl.List = append(bl.List, ret)
	setPos(bl, x.Pos())
	return bl
}

// computeBranchNext computes the branchNext numbering
// and determines which labels end up inside which range-over-func loop bodies.
func (r *rewriter) computeBranchNext() {
	if r.labelLoop != nil {
		return
	}

	r.labelLoop = make(map[string]*syntax.ForStmt)
	r.branchNext = make(map[branch]int)

	var labels []string
	var stack []syntax.Node
	var forStack []*syntax.ForStmt
	forStack = append(forStack, nil)
	syntax.Inspect(r.body, func(n syntax.Node) bool {
		if n != nil {
			stack = append(stack, n)
			if nfor, ok := forRangeFunc(n); ok {
				forStack = append(forStack, nfor)
			}
			if n, ok := n.(*syntax.LabeledStmt); ok {
				l := n.Label.Value
				labels = append(labels, l)
				f := forStack[len(forStack)-1]
				r.labelLoop[l] = f
			}
		} else {
			n := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			if n == forStack[len(forStack)-1] {
				forStack = forStack[:len(forStack)-1]
			}
		}
		return true
	})

	// Assign numbers to all the labels we observed.
	used := -2
	for _, l := range labels {
		used -= 3
		r.branchNext[branch{syntax.Break, l}] = used
		r.branchNext[branch{syntax.Continue, l}] = used + 1
		r.branchNext[branch{syntax.Goto, l}] = used + 2
	}
}

// endLoop finishes the conversion of a range-over-func loop.
// We have inspected and rewritten the body of the loop and can now
// construct the body function and rewrite the for loop into a call
// bracketed by any declarations and checks it requires.
func (r *rewriter) endLoop(loop *forLoop) {
	// Pick apart for range X { ... }
	nfor := loop.nfor
	start, end := nfor.Pos(), nfor.Body.Rbrace // start, end position of for loop
	rclause := nfor.Init.(*syntax.RangeClause)
	rfunc := types2.CoreType(rclause.X.GetTypeInfo().Type).(*types2.Signature) // type of X - func(func(...)bool)
	if rfunc.Params().Len() != 1 {
		base.Fatalf("invalid typecheck of range func")
	}

	ftyp := types2.AsSignature(rfunc.Params().At(0).Type()) // func(...) bool
	if ftyp.Results().Len() != 1 {
		base.Fatalf("invalid typecheck of range func")
	}

	// Build X(bodyFunc)
	call := &syntax.ExprStmt{
		X: &syntax.CallExpr{
			Fun: rclause.X,
			ArgList: []syntax.Expr{
				r.bodyFunc(nfor.Body.List, syntax.UnpackListExpr(rclause.Lhs), rclause.Def, ftyp, start, end),
			},
		},
	}
	setPos(call, start)

	// Build checks based on #next after X(bodyFunc)
	checks := r.checks(loop, end)

	// Rewrite for vars := range X { ... } to
	//
	//	{
	//		r.declStmt
	//		call
	//		checks
	//	}
	//
	// The r.declStmt can be added to by this loop or any inner loop
	// during the creation of r.bodyFunc; it is only emitted in the outermost
	// converted range loop.
	block := &syntax.BlockStmt{Rbrace: end}
	setPos(block, start)
	if len(r.forStack) == 1 && r.declStmt != nil {
		setPos(r.declStmt, start)
		block.List = append(block.List, r.declStmt)
	}

	// declare the exitFlag here so it has proper scope and zeroing
	if r.checkFuncMisuse() {
		exitFlagDecl := &syntax.DeclStmt{DeclList: []syntax.Decl{loop.exitFlagDecl}}
		block.List = append(block.List, exitFlagDecl)
	}

	// iteratorFunc(bodyFunc)
	block.List = append(block.List, call)

	if r.checkFuncMisuse() {
		// iteratorFunc has exited, mark the exit flag for the body
		block.List = append(block.List, r.setExited())
	}
	block.List = append(block.List, checks...)

	if len(r.forStack) == 1 { // ending an outermost loop
		r.declStmt = nil
		r.nextVar = nil
		r.retVars = nil
		r.defers = nil
	}

	r.rewritten[nfor] = block
}

func (r *rewriter) setExited() *syntax.AssignStmt {
	return r.setExitedAt(len(r.forStack) - 1)
}

func (r *rewriter) setExitedAt(index int) *syntax.AssignStmt {
	loop := r.forStack[index]
	return &syntax.AssignStmt{
		Lhs: r.useVar(loop.exitFlag),
		Rhs: r.useVar(r.true),
	}
}

// bodyFunc converts the loop body (control flow has already been updated)
// to a func literal that can be passed to the range function.
//
// vars is the range variables from the range statement.
// def indicates whether this is a := range statement.
// ftyp is the type of the function we are creating
// start and end are the syntax positions to use for new nodes
// that should be at the start or end of the loop.
func (r *rewriter) bodyFunc(body []syntax.Stmt, lhs []syntax.Expr, def bool, ftyp *types2.Signature, start, end syntax.Pos) *syntax.FuncLit {
	// Starting X(bodyFunc); build up bodyFunc first.
	var params, results []*types2.Var
	results = append(results, types2.NewVar(start, nil, "", r.bool.Type()))
	bodyFunc := &syntax.FuncLit{
		// Note: Type is ignored but needs to be non-nil to avoid panic in syntax.Inspect.
		Type: &syntax.FuncType{},
		Body: &syntax.BlockStmt{
			List:   []syntax.Stmt{},
			Rbrace: end,
		},
	}
	setPos(bodyFunc, start)

	for i := 0; i < ftyp.Params().Len(); i++ {
		typ := ftyp.Params().At(i).Type()
		var paramVar *types2.Var
		if i < len(lhs) && def {
			// Reuse range variable as parameter.
			x := lhs[i]
			paramVar = r.info.Defs[x.(*syntax.Name)].(*types2.Var)
		} else {
			// Declare new parameter and assign it to range expression.
			paramVar = types2.NewVar(start, r.pkg, fmt.Sprintf("#p%d", 1+i), typ)
			if i < len(lhs) {
				x := lhs[i]
				as := &syntax.AssignStmt{Lhs: x, Rhs: r.useVar(paramVar)}
				as.SetPos(x.Pos())
				setPos(as.Rhs, x.Pos())
				bodyFunc.Body.List = append(bodyFunc.Body.List, as)
			}
		}
		params = append(params, paramVar)
	}

	tv := syntax.TypeAndValue{
		Type: types2.NewSignatureType(nil, nil, nil,
			types2.NewTuple(params...),
			types2.NewTuple(results...),
			false),
	}
	tv.SetIsValue()
	bodyFunc.SetTypeInfo(tv)

	loop := r.forStack[len(r.forStack)-1]

	if r.checkFuncMisuse() {
		bodyFunc.Body.List = append(bodyFunc.Body.List, r.assertNotExited(start, loop))
	}

	// Original loop body (already rewritten by editStmt during inspect).
	bodyFunc.Body.List = append(bodyFunc.Body.List, body...)

	// return true to continue at end of loop body
	ret := &syntax.ReturnStmt{Results: r.useVar(r.true)}
	ret.SetPos(end)
	bodyFunc.Body.List = append(bodyFunc.Body.List, ret)

	return bodyFunc
}

// checks returns the post-call checks that need to be done for the given loop.
func (r *rewriter) checks(loop *forLoop, pos syntax.Pos) []syntax.Stmt {
	var list []syntax.Stmt
	if len(loop.checkBranch) > 0 {
		did := make(map[branch]bool)
		for _, br := range loop.checkBranch {
			if did[br] {
				continue
			}
			did[br] = true
			doBranch := &syntax.BranchStmt{Tok: br.tok, Label: &syntax.Name{Value: br.label}}
			list = append(list, r.ifNext(syntax.Eql, r.branchNext[br], doBranch))
		}
	}
	if len(r.forStack) == 1 {
		if loop.checkRetArgs {
			list = append(list, r.ifNext(syntax.Eql, -2, retStmt(r.useList(r.retVars))))
		}
		if loop.checkRet {
			list = append(list, r.ifNext(syntax.Eql, -1, retStmt(nil)))
		}
	} else {
		if loop.checkRetArgs || loop.checkRet {
			// Note: next < 0 also handles gotos handled by outer loops.
			// We set checkRet in that case to trigger this check.
			list = append(list, r.ifNext(syntax.Lss, 0, retStmt(r.useVar(r.false))))
		}
		if loop.checkBreak {
			list = append(list, r.ifNext(syntax.Geq, perLoopStep, retStmt(r.useVar(r.false))))
		}
		if loop.checkContinue {
			list = append(list, r.ifNext(syntax.Eql, perLoopStep-1, retStmt(r.useVar(r.true))))
		}
	}

	for _, j := range list {
		setPos(j, pos)
	}
	return list
}

// retStmt returns a return statement returning the given return values.
func retStmt(results syntax.Expr) *syntax.ReturnStmt {
	return &syntax.ReturnStmt{Results: results}
}

// ifNext returns the statement:
//
//	if #next op c { adjust; then }
//
// When op is >=, adjust is #next -= c.
// When op is == and c is not -1 or -2, adjust is #next = 0.
// Otherwise adjust is omitted.
func (r *rewriter) ifNext(op syntax.Operator, c int, then syntax.Stmt) syntax.Stmt {
	nif := &syntax.IfStmt{
		Cond: &syntax.Operation{Op: op, X: r.next(), Y: r.intConst(c)},
		Then: &syntax.BlockStmt{
			List: []syntax.Stmt{then},
		},
	}
	tv := syntax.TypeAndValue{Type: r.bool.Type()}
	tv.SetIsValue()
	nif.Cond.SetTypeInfo(tv)

	if op == syntax.Geq {
		sub := &syntax.AssignStmt{
			Op:  syntax.Sub,
			Lhs: r.next(),
			Rhs: r.intConst(c),
		}
		nif.Then.List = []syntax.Stmt{sub, then}
	}
	if op == syntax.Eql && c != -1 && c != -2 {
		clr := &syntax.AssignStmt{
			Lhs: r.next(),
			Rhs: r.intConst(0),
		}
		nif.Then.List = []syntax.Stmt{clr, then}
	}

	return nif
}

// setValueType marks x as a value with type typ.
func setValueType(x syntax.Expr, typ syntax.Type) {
	tv := syntax.TypeAndValue{Type: typ}
	tv.SetIsValue()
	x.SetTypeInfo(tv)
}

// assertNotExited returns the statement:
//
//	if #exitK { runtime.panicrangeexit() }
//
// where #exitK is the exit guard for loop.
func (r *rewriter) assertNotExited(start syntax.Pos, loop *forLoop) syntax.Stmt {
	callPanicExpr := &syntax.CallExpr{
		Fun: runtimeSym(r.info, "panicrangeexit"),
	}
	setValueType(callPanicExpr, nil) // no result type

	callPanic := &syntax.ExprStmt{X: callPanicExpr}

	nif := &syntax.IfStmt{
		Cond: r.useVar(loop.exitFlag),
		Then: &syntax.BlockStmt{
			List: []syntax.Stmt{callPanic},
		},
	}
	setPos(nif, start)
	return nif
}

// next returns a reference to the #next variable.
func (r *rewriter) next() *syntax.Name {
	if r.nextVar == nil {
		r.nextVar = r.declVar("#next", r.int.Type(), nil)
	}
	return r.useVar(r.nextVar)
}

// forRangeFunc checks whether n is a range-over-func.
// If so, it returns n.(*syntax.ForStmt), true.
// Otherwise it returns nil, false.
func forRangeFunc(n syntax.Node) (*syntax.ForStmt, bool) {
	nfor, ok := n.(*syntax.ForStmt)
	if !ok {
		return nil, false
	}
	nrange, ok := nfor.Init.(*syntax.RangeClause)
	if !ok {
		return nil, false
	}
	_, ok = types2.CoreType(nrange.X.GetTypeInfo().Type).(*types2.Signature)
	if !ok {
		return nil, false
	}
	return nfor, true
}

// intConst returns syntax for an integer literal with the given value.
func (r *rewriter) intConst(c int) *syntax.BasicLit {
	lit := &syntax.BasicLit{
		Value: fmt.Sprint(c),
		Kind:  syntax.IntLit,
	}
	tv := syntax.TypeAndValue{Type: r.int.Type(), Value: constant.MakeInt64(int64(c))}
	tv.SetIsValue()
	lit.SetTypeInfo(tv)
	return lit
}

// useVar returns syntax for a reference to decl, which should be its declaration.
func (r *rewriter) useVar(obj types2.Object) *syntax.Name {
	n := syntax.NewName(nopos, obj.Name())
	tv := syntax.TypeAndValue{Type: obj.Type()}
	tv.SetIsValue()
	n.SetTypeInfo(tv)
	r.info.Uses[n] = obj
	return n
}

// useList is useVar for a list of decls.
func (r *rewriter) useList(vars []types2.Object) syntax.Expr {
	var new []syntax.Expr
	for _, obj := range vars {
		new = append(new, r.useVar(obj))
	}
	if len(new) == 1 {
		return new[0]
	}
	return &syntax.ListExpr{ElemList: new}
}

// declVar declares a variable with a given name type and initializer value.
func (r *rewriter) declVar(name string, typ types2.Type, init syntax.Expr) *types2.Var {
	if r.declStmt == nil {
		r.declStmt = &syntax.DeclStmt{}
	}
	stmt := r.declStmt
	obj := types2.NewVar(stmt.Pos(), r.pkg, name, typ)
	n := syntax.NewName(stmt.Pos(), name)
	tv := syntax.TypeAndValue{Type: typ}
	tv.SetIsValue()
	n.SetTypeInfo(tv)
	r.info.Defs[n] = obj
	stmt.DeclList = append(stmt.DeclList, &syntax.VarDecl{
		NameList: []*syntax.Name{n},
		// Note: Type is ignored
		Values: init,
	})
	return obj
}

// declType declares a type with the given name and type.
// This is more like "type name = typ" than "type name typ".
func declType(pos syntax.Pos, name string, typ types2.Type) *syntax.Name {
	n := syntax.NewName(pos, name)
	n.SetTypeInfo(syntax.TypeAndValue{Type: typ})
	return n
}

// runtimePkg is a fake runtime package that contains what we need to refer to in package runtime.
var runtimePkg = func() *types2.Package {
	var nopos syntax.Pos
	pkg := types2.NewPackage("runtime", "runtime")
	anyType := types2.Universe.Lookup("any").Type()

	// func deferrangefunc() unsafe.Pointer
	obj := types2.NewFunc(nopos, pkg, "deferrangefunc", types2.NewSignatureType(nil, nil, nil, nil, types2.NewTuple(types2.NewParam(nopos, pkg, "extra", anyType)), false))
	pkg.Scope().Insert(obj)

	// func panicrangeexit()
	obj = types2.NewFunc(nopos, pkg, "panicrangeexit", types2.NewSignatureType(nil, nil, nil, nil, nil, false))
	pkg.Scope().Insert(obj)

	return pkg
}()

// runtimeSym returns a reference to a symbol in the fake runtime package.
func runtimeSym(info *types2.Info, name string) *syntax.Name {
	obj := runtimePkg.Scope().Lookup(name)
	n := syntax.NewName(nopos, "runtime."+name)
	tv := syntax.TypeAndValue{Type: obj.Type()}
	tv.SetIsValue()
	tv.SetIsRuntimeHelper()
	n.SetTypeInfo(tv)
	info.Uses[n] = obj
	return n
}

// setPos walks the top structure of x that has no position assigned
// and assigns it all to have position pos.
// When setPos encounters a syntax node with a position assigned,
// setPos does not look inside that node.
// setPos only needs to handle syntax we create in this package;
// all other syntax should have positions assigned already.
func setPos(x syntax.Node, pos syntax.Pos) {
	if x == nil {
		return
	}
	syntax.Inspect(x, func(n syntax.Node) bool {
		if n == nil || n.Pos() != nopos {
			return false
		}
		n.SetPos(pos)
		switch n := n.(type) {
		case *syntax.BlockStmt:
			if n.Rbrace == nopos {
				n.Rbrace = pos
			}
		}
		return true
	})
}
