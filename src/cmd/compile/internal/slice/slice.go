// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slice

// This file implements a stack-allocation optimization
// for the backing store of slices.
//
// Consider the code:
//
//     var s []int
//     for i := range ... {
//        s = append(s, i)
//     }
//     return s
//
// Some of the append operations will need to do an allocation
// by calling growslice. This will happen on the 1st, 2nd, 4th,
// 8th, etc. append calls. The allocations done by all but the
// last growslice call will then immediately be garbage.
//
// We'd like to avoid doing some of those intermediate
// allocations if possible.
//
// If we can determine that the "return s" statement is the
// *only* way that the backing store for s escapes, then we
// can rewrite the code to something like:
//
//     var s []int
//     for i := range N {
//        s = append(s, i)
//     }
//     s = move2heap(s)
//     return s
//
// Using the move2heap runtime function, which does:
//
//     move2heap(s):
//         If s is not backed by a stackframe-allocated
//         backing store, return s. Otherwise, copy s
//         to the heap and return the copy.
//
// Now we can treat the backing store of s allocated at the
// append site as not escaping. Previous stack allocation
// optimizations now apply, which can use a fixed-size
// stack-allocated backing store for s when appending.
// (See ../ssagen/ssa.go:(*state).append)
//
// It is tricky to do this optimization safely. To describe
// our analysis, we first define what an "exclusive" slice
// variable is.
//
// A slice variable (a variable of slice type) is called
// "exclusive" if, when it has a reference to a
// stackframe-allocated backing store, it is the only
// variable with such a reference.
//
// In other words, a slice variable is exclusive if
// any of the following holds:
//  1) It points to a heap-allocated backing store
//  2) It points to a stack-allocated backing store
//     for any parent frame.
//  3) It is the only variable that references its
//     backing store.
//  4) It is nil.
//
// The nice thing about exclusive slice variables is that
// it is always safe to do
//    s = move2heap(s)
// whenever s is an exclusive slice variable. Because no
// one else has a reference to the backing store, no one
// else can tell that we moved the backing store from one
// location to another.
//
// Note that exclusiveness is a dynamic property. A slice
// variable may be exclusive during some parts of execution
// and not exclusive during others.
//
// The following operations set or preserve the exclusivity
// of a slice variable s:
//     s = nil
//     s = append(s, ...)
//     s = s[i:j]
//     ... = s[i]
//     s[i] = ...
//     f(s) where f does not escape its argument
// Other operations destroy exclusivity. A non-exhaustive list includes:
//     x = s
//     *p = s
//     f(s) where f escapes its argument
//     return s
// To err on the safe side, we white list exclusivity-preserving
// operations and we asssume that any other operations that mention s
// destroy its exclusivity.
//
// Our strategy is to move the backing store of s to the heap before
// any exclusive->nonexclusive transition. That way, s will only ever
// have a reference to a stack backing store while it is exclusive.
//
// move2heap for a variable s is implemented with:
//     if s points to within the stack frame {
//         s2 := make([]T, s.len, s.cap)
//         copy(s2[:s.cap], s[:s.cap])
//         s = s2
//     }
// Note that in general we need to copy all of s[:cap(s)] elements when
// moving to the heap. As an optimization, we keep track of slice variables
// whose capacity, and the elements in s[len(s):cap(s)], are never accessed.
// For those slice variables, we can allocate to the next size class above
// the length, which saves memory and copying cost.

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
)

func Funcs(all []*ir.Func) {
	if base.Flag.N != 0 {
		return
	}
	for _, fn := range all {
		analyze(fn)
	}
}

func analyze(fn *ir.Func) {
	type sliceInfo struct {
		// Slice variable.
		s *ir.Name

		// Count of uses that this pass understands.
		okUses int32
		// Count of all uses found.
		allUses int32

		// A place where the slice variable transitions from
		// exclusive to nonexclusive.
		// We could keep track of more than one, but one is enough for now.
		// Currently, this can be either a return statement or
		// an assignment.
		// TODO: other possible transitions?
		transition ir.Stmt

		// Each s = append(s, ...) instance we found.
		appends []*ir.CallExpr

		// Weight of the number of s = append(s, ...) instances we found.
		// The optimizations we do are only really useful if there are at
		// least weight 2. (Note: appends in loops have weight >= 2.)
		appendWeight int

		// Loop depth at declaration point.
		// Use for heuristics only, it is not guaranteed to be correct
		// in the presence of gotos.
		declDepth int

		// Whether we ever do cap(s), or other operations that use cap(s)
		// (possibly implicitly), like s[i:j].
		capUsed bool
	}

	// Every variable (*ir.Name) that we are tracking will have
	// a non-nil *sliceInfo in its Opt field.
	haveLocalSlice := false
	maxStackSize := int64(base.Debug.VariableMakeThreshold)
	var namedRets []*ir.Name
	for _, s := range fn.Dcl {
		if !s.Type().IsSlice() {
			continue
		}
		if s.Type().Elem().Size() > maxStackSize {
			continue
		}
		if !base.VariableMakeHash.MatchPos(s.Pos(), nil) {
			continue
		}
		s.Opt = &sliceInfo{s: s} // start tracking s
		haveLocalSlice = true
		if s.Class == ir.PPARAMOUT {
			namedRets = append(namedRets, s)
		}
	}
	if !haveLocalSlice {
		return
	}

	// Keep track of loop depth while walking.
	loopDepth := 0

	// tracking returns the info for the slice variable if n is a slice
	// variable that we're still considering, or nil otherwise.
	tracking := func(n ir.Node) *sliceInfo {
		if n == nil || n.Op() != ir.ONAME {
			return nil
		}
		s := n.(*ir.Name)
		if s.Opt == nil {
			return nil
		}
		return s.Opt.(*sliceInfo)
	}

	// addTransition(n, loc) records that s experiences an exclusive->nonexclusive
	// transition somewhere within loc.
	addTransition := func(i *sliceInfo, loc ir.Stmt) {
		if i.transition != nil {
			// We only keep track of a single exclusive->nonexclusive transition
			// for a slice variable. If we find more than one, give up.
			// (More than one transition location would be fine, but we would
			// start to get worried about introducing too much additional code.)
			i.s.Opt = nil
			return
		}
		if loopDepth > i.declDepth {
			// Conservatively, we disable this optimization when the
			// transition is inside a loop. This can result in adding
			// overhead unnecessarily in cases like:
			// func f(n int, p *[]byte) {
			//     var s []byte
			//     for i := range n {
			//         *p = s
			//         s = append(s, 0)
			//     }
			// }
			i.s.Opt = nil
			return
		}
		i.transition = loc
	}

	// Examine an x = y assignment that occurs somewhere within statement stmt.
	assign := func(x, y ir.Node, stmt ir.Stmt) {
		if i := tracking(x); i != nil {
			// s = y. Check for understood patterns for y.
			if y == nil || y.Op() == ir.ONIL {
				// s = nil is ok.
				i.okUses++
			} else if y.Op() == ir.OSLICELIT {
				// s = []{...} is ok.
				// Note: this reveals capacity. Should it?
				i.okUses++
				i.capUsed = true
			} else if y.Op() == ir.OSLICE {
				y := y.(*ir.SliceExpr)
				if y.X == i.s {
					// s = s[...:...] is ok
					i.okUses += 2
					i.capUsed = true
				}
			} else if y.Op() == ir.OAPPEND {
				y := y.(*ir.CallExpr)
				if y.Args[0] == i.s {
					// s = append(s, ...) is ok
					i.okUses += 2
					i.appends = append(i.appends, y)
					i.appendWeight += 1 + (loopDepth - i.declDepth)
				}
				// TODO: s = append(nil, ...)?
			}
			// Note that technically s = make([]T, ...) preserves exclusivity, but
			// we don't track that because we assume users who wrote that know
			// better than the compiler does.

			// TODO: figure out how to handle s = fn(..., s, ...)
			// It would be nice to maintain exclusivity of s in this situation.
			// But unfortunately, fn can return one of its other arguments, which
			// may be a slice with a stack-allocated backing store other than s.
			// (which may have preexisting references to its backing store).
			//
			// Maybe we could do it if s is the only argument?
		}

		if i := tracking(y); i != nil {
			// ... = s
			// Treat this as an exclusive->nonexclusive transition.
			i.okUses++
			addTransition(i, stmt)
		}
	}

	var do func(ir.Node) bool
	do = func(n ir.Node) bool {
		if n == nil {
			return false
		}
		switch n.Op() {
		case ir.ONAME:
			if i := tracking(n); i != nil {
				// A use of a slice variable. Count it.
				i.allUses++
			}
		case ir.ODCL:
			n := n.(*ir.Decl)
			if i := tracking(n.X); i != nil {
				i.okUses++
				i.declDepth = loopDepth
			}
		case ir.OINDEX:
			n := n.(*ir.IndexExpr)
			if i := tracking(n.X); i != nil {
				// s[i] is ok.
				i.okUses++
			}
		case ir.OLEN:
			n := n.(*ir.UnaryExpr)
			if i := tracking(n.X); i != nil {
				// len(s) is ok
				i.okUses++
			}
		case ir.OCAP:
			n := n.(*ir.UnaryExpr)
			if i := tracking(n.X); i != nil {
				// cap(s) is ok
				i.okUses++
				i.capUsed = true
			}
		case ir.OADDR:
			n := n.(*ir.AddrExpr)
			if n.X.Op() == ir.OINDEX {
				n := n.X.(*ir.IndexExpr)
				if i := tracking(n.X); i != nil {
					// &s[i] is definitely a nonexclusive transition.
					// (We need this case because s[i] is ok, but &s[i] is not.)
					i.s.Opt = nil
				}
			}
		case ir.ORETURN:
			n := n.(*ir.ReturnStmt)
			for _, x := range n.Results {
				if i := tracking(x); i != nil {
					i.okUses++
					// We go exclusive->nonexclusive here
					addTransition(i, n)
				}
			}
			if len(n.Results) == 0 {
				// Uses of named result variables are implicit here.
				for _, x := range namedRets {
					if i := tracking(x); i != nil {
						addTransition(i, n)
					}
				}
			}
		case ir.OCALLFUNC:
			n := n.(*ir.CallExpr)
			for idx, arg := range n.Args {
				if i := tracking(arg); i != nil {
					if !argLeak(n, idx) {
						// Passing s to a nonescaping arg is ok.
						i.okUses++
						i.capUsed = true
					}
				}
			}
		case ir.ORANGE:
			// Range over slice is ok.
			n := n.(*ir.RangeStmt)
			if i := tracking(n.X); i != nil {
				i.okUses++
			}
		case ir.OAS:
			n := n.(*ir.AssignStmt)
			assign(n.X, n.Y, n)
		case ir.OAS2:
			n := n.(*ir.AssignListStmt)
			for i := range len(n.Lhs) {
				assign(n.Lhs[i], n.Rhs[i], n)
			}
		case ir.OCLOSURE:
			n := n.(*ir.ClosureExpr)
			for _, v := range n.Func.ClosureVars {
				do(v.Outer)
			}
		}
		if n.Op() == ir.OFOR || n.Op() == ir.ORANGE {
			// Note: loopDepth isn't really right for init portion
			// of the for statement, but that's ok. Correctness
			// does not depend on depth info.
			loopDepth++
			defer func() { loopDepth-- }()
		}
		// Check all the children.
		ir.DoChildren(n, do)
		return false
	}

	// Run the analysis over the whole body.
	for _, stmt := range fn.Body {
		do(stmt)
	}

	// Process accumulated info to find slice variables
	// that we can allocate on the stack.
	for _, s := range fn.Dcl {
		if s.Opt == nil {
			continue
		}
		i := s.Opt.(*sliceInfo)
		s.Opt = nil
		if i.okUses != i.allUses {
			// Some use of i.s that don't understand lurks. Give up.
			continue
		}

		// At this point, we've decided that we *can* do
		// the optimization.

		if i.transition == nil {
			// Exclusive for its whole lifetime. That means it
			// didn't escape. We can already handle nonescaping
			// slices without this pass.
			continue
		}
		if i.appendWeight < 2 {
			// This optimization only really helps if there is
			// (dynamically) more than one append.
			continue
		}

		// Commit point - at this point we've decided we *should*
		// do the optimization.

		// Insert a move2heap operation before the exclusive->nonexclusive
		// transition.
		move := ir.NewMoveToHeapExpr(i.transition.Pos(), i.s)
		if i.capUsed {
			move.PreserveCapacity = true
		}
		move.RType = reflectdata.AppendElemRType(i.transition.Pos(), i.appends[0])
		move.SetType(i.s.Type())
		move.SetTypecheck(1)
		as := ir.NewAssignStmt(i.transition.Pos(), i.s, move)
		as.SetTypecheck(1)
		i.transition.PtrInit().Prepend(as)
		// Note: we prepend because we need to put the move2heap
		// operation first, before any other init work, as the transition
		// might occur in the init work.

		// Now that we've inserted a move2heap operation before every
		// exclusive -> nonexclusive transition, appends can now use
		// stack backing stores.
		// (This is the whole point of this pass, to enable stack
		// allocation of append backing stores.)
		for _, a := range i.appends {
			a.SetEsc(ir.EscNone)
			if i.capUsed {
				a.UseBuf = true
			}
		}
	}
}

// argLeak reports if the idx'th argument to the call n escapes anywhere
// (to the heap, another argument, return value, etc.)
// If unknown returns true.
func argLeak(n *ir.CallExpr, idx int) bool {
	if n.Op() != ir.OCALLFUNC {
		return true
	}
	fn := ir.StaticCalleeName(ir.StaticValue(n.Fun))
	if fn == nil {
		return true
	}
	fntype := fn.Type()
	if recv := fntype.Recv(); recv != nil {
		if idx == 0 {
			return escape.ParseLeaks(recv.Note).Any()
		}
		idx--
	}
	return escape.ParseLeaks(fntype.Params()[idx].Note).Any()
}
