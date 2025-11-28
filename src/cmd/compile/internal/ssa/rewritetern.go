// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"internal/goarch"
	"slices"
)

var truthTableValues [3]uint8 = [3]uint8{0b1111_0000, 0b1100_1100, 0b1010_1010}

func (slop SIMDLogicalOP) String() string {
	if slop == sloInterior {
		return "leaf"
	}
	interior := ""
	if slop&sloInterior != 0 {
		interior = "+interior"
	}
	switch slop &^ sloInterior {
	case sloAnd:
		return "and" + interior
	case sloXor:
		return "xor" + interior
	case sloOr:
		return "or" + interior
	case sloAndNot:
		return "andNot" + interior
	case sloNot:
		return "not" + interior
	}
	return "wrong"
}

func rewriteTern(f *Func) {
	if f.maxCPUFeatures == CPUNone {
		return
	}

	arch := f.Config.Ctxt().Arch.Family
	// TODO there are other SIMD architectures
	if arch != goarch.AMD64 {
		return
	}

	boolExprTrees := make(map[*Value]SIMDLogicalOP)

	// Find logical-expr expression trees, including leaves.
	// interior nodes will be marked sloInterior,
	// root nodes will not be marked sloInterior,
	// leaf nodes are only marked sloInterior.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			slo := classifyBooleanSIMD(v)
			switch slo {
			case sloOr,
				sloAndNot,
				sloXor,
				sloAnd:
				boolExprTrees[v.Args[1]] |= sloInterior
				fallthrough
			case sloNot:
				boolExprTrees[v.Args[0]] |= sloInterior
				boolExprTrees[v] |= slo
			}
		}
	}

	// get a canonical sorted set of roots
	var roots []*Value
	for v, slo := range boolExprTrees {
		if f.pass.debug > 1 {
			f.Warnl(v.Pos, "%s has SLO %v", v.LongString(), slo)
		}

		if slo&sloInterior == 0 && v.Block.CPUfeatures.hasFeature(CPUavx512) {
			roots = append(roots, v)
		}
	}
	slices.SortFunc(roots, func(u, v *Value) int { return int(u.ID - v.ID) }) // IDs are small enough to not care about overflow.

	// This rewrite works by iterating over the root set.
	// For each boolean expression, it walks the expression
	// bottom up accumulating sets of variables mentioned in
	// subexpressions, lazy-greedily finding the largest subexpressions
	// of 3 inputs that can be rewritten to use ternary-truth-table instructions.

	// rewrite recursively attempts to replace v and v's subexpressions with
	// ternary-logic truth-table operations, returning a set of not more than 3
	// subexpressions within v that may be combined into a parent's replacement.
	// V need not have the CPU features that allow a ternary-logic operation;
	// in that case, v will not be rewritten.  Replacements also require
	// exactly 3 different variable inputs to a boolean expression.
	//
	// Given the CPU feature and 3 inputs, v is replaced in the following
	// cases:
	//
	// 1) v is a root
	// 2) u = NOT(v) and u lacks the CPU feature
	// 3) u = OP(v, w) and u lacks the CPU feature
	// 4) u = OP(v, w) and u has more than 3 variable inputs.	var rewrite func(v *Value) [3]*Value
	var rewrite func(v *Value) [3]*Value

	// computeTT returns the truth table for a boolean expression
	// over the variables in vars, where vars[0] varies slowest in
	// the truth table and vars[2] varies fastest.
	// e.g. computeTT( "and(x, or(y, not(z)))", {x,y,z} ) returns
	// (bit 0 first) 0 0 0 0 1 0 1 1 = (reversed) 1101_0000 = 0xD0
	//            x: 0 0 0 0 1 1 1 1
	//            y: 0 0 1 1 0 0 1 1
	//            z: 0 1 0 1 0 1 0 1
	var computeTT func(v *Value, vars [3]*Value) uint8

	// combine two sets of variables into one, returning ok/not
	// if the two sets contained 3 or fewer elements.  Combine
	// ensures that the sets of Values never contain duplicates.
	// (Duplicates would create less-efficient code, not incorrect code.)
	combine := func(a, b [3]*Value) ([3]*Value, bool) {
		var c [3]*Value
		i := 0
		for _, v := range a {
			if v == nil {
				break
			}
			c[i] = v
			i++
		}
	bloop:
		for _, v := range b {
			if v == nil {
				break
			}
			for _, u := range a {
				if v == u {
					continue bloop
				}
			}
			if i == 3 {
				return [3]*Value{}, false
			}
			c[i] = v
			i++
		}
		return c, true
	}

	computeTT = func(v *Value, vars [3]*Value) uint8 {
		i := 0
		for ; i < len(vars); i++ {
			if vars[i] == v {
				return truthTableValues[i]
			}
		}
		slo := boolExprTrees[v] &^ sloInterior
		a := computeTT(v.Args[0], vars)
		switch slo {
		case sloNot:
			return ^a
		case sloAnd:
			return a & computeTT(v.Args[1], vars)
		case sloXor:
			return a ^ computeTT(v.Args[1], vars)
		case sloOr:
			return a | computeTT(v.Args[1], vars)
		case sloAndNot:
			return a & ^computeTT(v.Args[1], vars)
		}
		panic("switch should have covered all cases, or unknown var in logical expression")
	}

	replace := func(a0 *Value, vars0 [3]*Value) {
		imm := computeTT(a0, vars0)
		op := ternOpForLogical(a0.Op)
		if op == a0.Op {
			panic(fmt.Errorf("should have mapped away from input op, a0 is %s", a0.LongString()))
		}
		if f.pass.debug > 0 {
			f.Warnl(a0.Pos, "Rewriting %s into %v of 0b%b %v %v %v", a0.LongString(), op, imm,
				vars0[0], vars0[1], vars0[2])
		}
		a0.reset(op)
		a0.SetArgs3(vars0[0], vars0[1], vars0[2])
		a0.AuxInt = int64(int8(imm))
	}

	// addOne ensures the no-duplicates addition of a single value
	// to a set that is not full.  It seems possible that a shared
	// subexpression in tricky combination with blocks lacking the
	// AVX512 feature might permit this.
	addOne := func(vars [3]*Value, v *Value) [3]*Value {
		if vars[2] != nil {
			panic("rewriteTern.addOne, vars[2] should be nil")
		}
		if v == vars[0] || v == vars[1] {
			return vars
		}
		if vars[1] == nil {
			vars[1] = v
		} else {
			vars[2] = v
		}
		return vars
	}

	rewrite = func(v *Value) [3]*Value {
		slo := boolExprTrees[v]
		if slo == sloInterior { // leaf node, i.e., a "variable"
			return [3]*Value{v, nil, nil}
		}
		var vars [3]*Value
		hasFeature := v.Block.CPUfeatures.hasFeature(CPUavx512)
		if slo&sloNot == sloNot {
			vars = rewrite(v.Args[0])
			if !hasFeature {
				if vars[2] != nil {
					replace(v.Args[0], vars)
					return [3]*Value{v, nil, nil}
				}
				return vars
			}
		} else {
			var ok bool
			a0, a1 := v.Args[0], v.Args[1]
			vars0 := rewrite(a0)
			vars1 := rewrite(a1)
			vars, ok = combine(vars0, vars1)

			if f.pass.debug > 1 {
				f.Warnl(a0.Pos, "combine(%v, %v) -> %v, %v", vars0, vars1, vars, ok)
			}

			if !(ok && v.Block.CPUfeatures.hasFeature(CPUavx512)) {
				// too many variables, or cannot rewrite current values.
				// rewrite one or both subtrees if possible
				if vars0[2] != nil && a0.Block.CPUfeatures.hasFeature(CPUavx512) {
					replace(a0, vars0)
				}
				if vars1[2] != nil && a1.Block.CPUfeatures.hasFeature(CPUavx512) {
					replace(a1, vars1)
				}

				// 3-element var arrays are either rewritten, or unable to be rewritten
				// because of the features in effect in their block.  Either way, they
				// are treated as a "new var" if 3 elements are present.

				if vars0[2] == nil {
					if vars1[2] == nil {
						// both subtrees are 2-element and were not rewritten.
						//
						// TODO a clever person would look at subtrees of inputs,
						// e.g. rewrite
						//        ((a AND b) XOR b) XOR (d  XOR (c AND d))
						// to    (((a AND b) XOR b) XOR  d) XOR (c AND d)
						// to v = TERNLOG(truthtable, a, b, d) XOR (c AND d)
						// and return the variable set {v, c, d}
						//
						// But for now, just restart with a0 and a1.
						return [3]*Value{a0, a1, nil}
					} else {
						// a1 (maybe) rewrote, a0 has room for another var
						vars = addOne(vars0, a1)
					}
				} else if vars1[2] == nil {
					// a0 (maybe) rewrote, a1 has room for another var
					vars = addOne(vars1, a0)
				} else if !ok {
					// both (maybe) rewrote
					// a0 and a1 are different because otherwise their variable
					// sets would have combined "ok".
					return [3]*Value{a0, a1, nil}
				}
				// continue with either the vars from "ok" or the updated set of vars.
			}
		}
		// if root and 3 vars and hasFeature, rewrite.
		if slo&sloInterior == 0 && vars[2] != nil && hasFeature {
			replace(v, vars)
			return [3]*Value{v, nil, nil}
		}
		return vars
	}

	for _, v := range roots {
		if f.pass.debug > 1 {
			f.Warnl(v.Pos, "SLO root %s", v.LongString())
		}
		rewrite(v)
	}
}
