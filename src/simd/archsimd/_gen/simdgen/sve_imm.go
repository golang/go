// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"slices"
	"strings"
)

// sveImmRanges maps the ISA's description of an SVE immediate, as the loader
// passes it on in ImmDesc, to the immediate's range for an operation on
// esize-bit elements. The ISA states the range only in prose, and some ranges
// depend on the element width ("in the range 0 to number of bits per element
// minus 1"), so each description an operation uses is entered here by hand.
var sveImmRanges = map[string]func(esize int) (min, max int){
	// LSL (immediate).
	`Is the immediate shift amount, in the range 0 to number of bits per element minus 1, encoded in "tszh:tszl:imm3".`: func(esize int) (int, int) { return 0, esize - 1 },
}

// resolveSVEImmRanges sets the range of every SVE immediate in ops from
// sveImmRanges. It fails on an immediate whose description has no entry, and
// on an immediate of an exported API with a nonzero minimum, which the
// intrinsics' range check cannot enforce.
func resolveSVEImmRanges(ops []Operation) error {
	if !CurrentArch().isSVE() {
		return nil
	}
	var errs []string
	for i := range ops {
		op := &ops[i]
		for j := range op.In {
			in := &op.In[j]
			if in.Class != "immediate" || in.ImmDesc == nil {
				continue
			}
			r, ok := sveImmRanges[*in.ImmDesc]
			if !ok {
				errs = append(errs, fmt.Sprintf("%s %s: no entry in sveImmRanges for %q", op.Go, op.Asm, *in.ImmDesc))
				continue
			}
			lo, hi := r(sveImmElemBits(*op))
			if lo != 0 && (op.NoTypes == nil || *op.NoTypes != "true") {
				errs = append(errs, fmt.Sprintf("%s %s: immediate range %d to %d does not start at 0, which an API's range check cannot enforce", op.Go, op.Asm, lo, hi))
				continue
			}
			in.ImmMin, in.ImmMax = &lo, &hi
		}
	}
	if len(errs) > 0 {
		slices.Sort(errs)
		return fmt.Errorf("simdgen: SVE immediates:\n\t%s", strings.Join(slices.Compact(errs), "\n\t"))
	}
	return nil
}

// sveImmElemBits returns the element width an SVE immediate's range is stated
// in terms of: that of the operation's first vector operand.
func sveImmElemBits(op Operation) int {
	for _, o := range append(slices.Clone(op.In), op.Out...) {
		if o.Class == "vreg" && o.ElemBits != nil {
			return *o.ElemBits
		}
	}
	return 0
}
