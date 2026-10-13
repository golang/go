// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"testing"

	"simd/archsimd/_gen/simdgen/types"
)

func TestResolveSVEImmRanges(t *testing.T) {
	saved := *FlagArch
	*FlagArch = "sve"
	defer func() { *FlagArch = saved }()
	const desc = "test: in the range 0 to number of bits per element minus 1"
	sveImmRanges[desc] = func(esize int) (int, int) { return 0, esize - 1 }
	defer delete(sveImmRanges, desc)
	const descFrom1 = "test: in the range 1 to number of bits per element"
	sveImmRanges[descFrom1] = func(esize int) (int, int) { return 1, esize }
	defer delete(sveImmRanges, descFrom1)

	op := func(desc string, noTypes bool) Operation {
		var o Operation
		o.Go, o.Asm = "shift", "ZTEST"
		if noTypes {
			o.NoTypes = new("true")
		}
		o.In = []types.Operand{
			{Class: "immediate", ImmDesc: new(desc)},
			{Class: "vreg", ElemBits: new(16)},
		}
		o.Out = []types.Operand{{Class: "vreg", ElemBits: new(16)}}
		return o
	}

	ops := []Operation{op(desc, false), op(descFrom1, true)}
	if err := resolveSVEImmRanges(ops); err != nil {
		t.Fatal(err)
	}
	for i, want := range [][2]int{{0, 15}, {1, 16}} {
		in := ops[i].In[0]
		if in.ImmMin == nil || in.ImmMax == nil || *in.ImmMin != want[0] || *in.ImmMax != want[1] {
			t.Errorf("op %d: range %v to %v, want %d to %d", i, in.ImmMin, in.ImmMax, want[0], want[1])
		}
	}

	for _, c := range []struct {
		op   Operation
		want string
	}{
		{op("test: no entry", true), "no entry in sveImmRanges"},
		{op(descFrom1, false), "does not start at 0"},
	} {
		err := resolveSVEImmRanges([]Operation{c.op})
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("resolveSVEImmRanges(%q) = %v, want an error containing %q", *c.op.In[0].ImmDesc, err, c.want)
		}
	}
}
