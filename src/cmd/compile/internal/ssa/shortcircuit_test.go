// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestShortCircuit(t *testing.T) {
	c := testConfig(t)

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("arg1", OpArg, c.config.Types.Int64, 0, nil),
			Valu("arg2", OpArg, c.config.Types.Int64, 0, nil),
			Valu("arg3", OpArg, c.config.Types.Int64, 0, nil),
			Goto("b1")),
		Bloc("b1",
			Valu("cmp1", OpLess64, c.config.Types.Bool, 0, nil, "arg1", "arg2"),
			If("cmp1", "b2", "b3")),
		Bloc("b2",
			Valu("cmp2", OpLess64, c.config.Types.Bool, 0, nil, "arg2", "arg3"),
			Goto("b3")),
		Bloc("b3",
			Valu("phi2", OpPhi, c.config.Types.Bool, 0, nil, "cmp1", "cmp2"),
			If("phi2", "b4", "b5")),
		Bloc("b4",
			Valu("cmp3", OpLess64, c.config.Types.Bool, 0, nil, "arg3", "arg1"),
			Goto("b5")),
		Bloc("b5",
			Valu("phi3", OpPhi, c.config.Types.Bool, 0, nil, "phi2", "cmp3"),
			If("phi3", "b6", "b7")),
		Bloc("b6",
			Exit("mem")),
		Bloc("b7",
			Exit("mem")))

	CheckFunc(fun.f)
	shortcircuit(fun.f)
	CheckFunc(fun.f)

	for _, b := range fun.f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpPhi {
				t.Errorf("phi %s remains", v)
			}
		}
	}
}
