// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

// maybeRewriteLoopToDownwardCountingLoop tries to rewrite the loop to a
// downward counting loop checking against start if the loop body does
// not depend on ind or nxt and end is known before the loop.
// That means this code:
//
//	loop:
//		ind = (Phi (Const [x]) nxt),
//		if ind < end
//		then goto enter_loop
//		else goto exit_loop
//
//	enter_loop:
//		do something without using ind nor nxt
//		nxt = inc + ind
//		goto loop
//
//	exit_loop:
//
// is rewritten to:
//
//	loop:
//		ind = (Phi end nxt)
//		if (Const [x]) < ind
//		then goto enter_loop
//		else goto exit_loop
//
//	enter_loop:
//		do something without using ind nor nxt
//		nxt = ind - inc
//		goto loop
//
//	exit_loop:
//
// This is better because it only requires to keep ind then nxt alive while looping,
// while the original form keeps ind then nxt and end alive.
//
// If the loop could not be rewritten, it is left unchanged.
func maybeRewriteLoopToDownwardCountingLoop(f *Func, v indVar) {
	ind := v.ind
	nxt := v.nxt
	if !(ind.Uses == 2 && // 2 used by comparison and next
		nxt.Uses == 1) { // 1 used by induction
		return
	}

	start, end := v.min, v.max

	if !start.isGenericIntConst() {
		// if start is not a constant we would be winning nothing from inverting the loop
		return
	}
	if end.isGenericIntConst() {
		// TODO: if both start and end are constants we should rewrite such that the comparison
		// is against zero and nxt is ++ or -- operation
		// That means:
		//	for i := 2; i < 11; i += 2 {
		// should be rewritten to:
		//	for i := 5; 0 < i; i-- {
		return
	}

	if end.Block == ind.Block {
		// we can't rewrite loops where the condition depends on the loop body
		// this simple check is forced to work because if this is true a Phi in ind.Block must exist
		return
	}

	check := v.entry.Preds[0].b.Controls[0]

	neededRoom := -v.step

	// The whole range of safe numbers to land in to stop the loop is shifted by one if the bounds are exclusive.
	if neededRoom < 0 && v.flags&indVarMinExc == 1 {
		neededRoom++ // safe because it is always against the number's sign
	}
	if neededRoom > 0 && v.flags&indVarMaxInc == 0 {
		neededRoom-- // safe because it is always against the number's sign
	}

	switch check.Op {
	case OpLess8, OpLess16, OpLess32, OpLess64, OpLeq8, OpLeq16, OpLeq32, OpLeq64:
		if _, ok := safeAdd(start.AuxInt, neededRoom, uint(start.Type.Size())*8); !ok {
			// We lack sufficient room after start to safely land without an overflow.
			// See go.dev/issue/78303
			return
		}
	case OpLess8U, OpLess16U, OpLess32U, OpLess64U, OpLeq8U, OpLeq16U, OpLeq32U, OpLeq64U:
		panic(`parseIndVar didn't yet support unsigned induction variables, this code doesn't yet support them either.
If you are seeing this it is probably because you've fixed https://go.dev/issue/65918.
You need to update this code and add tests then.`)
	case OpEq8, OpEq16, OpEq32, OpEq64, OpNeq8, OpNeq16, OpNeq32, OpNeq64:
		panic(`parseIndVar didn't yet support induction variables using == or !=.
If you are seeing this it is probably because you've added support for them.
You need to update this code and add tests then.`)
	default:
		panic(fmt.Sprintf("unreachable; unexpected induction variable comparator %v %v", check, check.Op))
	}

	idxEnd, idxStart := -1, -1
	for i, v := range check.Args {
		if v == end {
			idxEnd = i
			break
		}
	}
	for i, v := range ind.Args {
		if v == start {
			idxStart = i
			break
		}
	}
	if idxEnd < 0 || idxStart < 0 {
		return
	}

	sdom := f.Sdom()
	// the end may not dominate the ind after rewrite, check it first
	if !sdom.IsAncestorEq(end.Block, ind.Block) {
		return
	}

	// swap start and end in the loop
	check.SetArg(idxEnd, start)
	ind.SetArg(idxStart, end)

	// invert the check
	check.Args[0], check.Args[1] = check.Args[1], check.Args[0]

	if nxt.Args[0] != ind {
		// unlike additions subtractions are not commutative so be sure we get it right
		nxt.Args[0], nxt.Args[1] = nxt.Args[1], nxt.Args[0]
	}

	switch nxt.Op {
	case OpAdd8:
		nxt.Op = OpSub8
	case OpAdd16:
		nxt.Op = OpSub16
	case OpAdd32:
		nxt.Op = OpSub32
	case OpAdd64:
		nxt.Op = OpSub64
	case OpSub8:
		nxt.Op = OpAdd8
	case OpSub16:
		nxt.Op = OpAdd16
	case OpSub32:
		nxt.Op = OpAdd32
	case OpSub64:
		nxt.Op = OpAdd64
	default:
		panic("unreachable")
	}

	if f.pass.debug > 0 {
		f.Warnl(ind.Pos, "Inverted loop iteration")
	}
}
