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
	if v.flags&indVarCountDown != 0 {
		start, end = end, start
	}

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

	check := ind.Block.Controls[0]
	// invert the check
	check.Args[0], check.Args[1] = check.Args[1], check.Args[0]

	// swap start and end in the loop
	for i, v := range check.Args {
		if v != end {
			continue
		}

		check.SetArg(i, start)
		goto replacedEnd
	}
	panic(fmt.Sprintf("unreachable, ind: %v, start: %v, end: %v", ind, start, end))
replacedEnd:

	for i, v := range ind.Args {
		if v != start {
			continue
		}

		ind.SetArg(i, end)
		goto replacedStart
	}
	panic(fmt.Sprintf("unreachable, ind: %v, start: %v, end: %v", ind, start, end))
replacedStart:

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
