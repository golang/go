// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"

	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

func TestPhioptSpecialCase(t *testing.T) {
	for _, layout := range []struct {
		name         string
		reversePreds bool
		directTrue   bool
	}{
		{"pred0/true", false, true},
		{"pred0/false", false, false},
		{"pred1/true", true, true},
		{"pred1/false", true, false},
	} {
		for _, cyclic := range []bool{false, true} {
			name := "acyclic/"
			if cyclic {
				name = "loop/"
			}
			t.Run(name+layout.name, func(t *testing.T) {
				c := testConfig(t)
				boolType := c.config.Types.Bool
				succ0, succ1 := "merge", "other"
				if !layout.directTrue {
					succ0, succ1 = succ1, succ0
				}
				arg0, arg1 := "false", "true"
				if layout.reversePreds {
					arg0, arg1 = arg1, arg0
				}
				mergeSucc := "done"
				if cyclic {
					mergeSucc = "other"
				}
				blocks := []bloc{
					Bloc("entry",
						Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
						Valu("cond", ssaop.OpArg, boolType, 0, nil),
						Valu("cond2", ssaop.OpArg, boolType, 0, nil),
						Valu("false", ssaop.OpConstBool, boolType, 0, nil),
						Valu("true", ssaop.OpConstBool, boolType, 1, nil),
						If("cond", succ0, succ1)),
					Bloc("other", If("cond2", "back", "exit")),
					Bloc("back", Goto("merge")),
					Bloc("merge",
						Valu("phi", ssaop.OpPhi, boolType, 0, nil, arg0, arg1),
						If("phi", mergeSucc, "exit")),
					Bloc("exit", Exit("mem")),
				}
				if !cyclic {
					blocks = append(blocks, Bloc("done", Exit("mem")))
				}
				if layout.reversePreds {
					blocks[0], blocks[2] = blocks[2], blocks[0]
				}
				fun := c.Fun("entry", blocks...)

				// The other successor dominates back in both graphs. In the
				// cyclic graph it can also be reached through merge, so cond
				// does not determine which predecessor supplies phi's value.
				CheckFunc(fun.f)
				phiopt(fun.f)
				CheckFunc(fun.f)

				v := fun.values["phi"]
				want := ssaop.OpCopy
				if layout.directTrue {
					want = ssaop.OpNot
				}
				if cyclic {
					want = ssaop.OpPhi
				}
				if v.Op != want {
					t.Fatalf("phi became %s, want %s", v.LongString(), want)
				}
				if cyclic {
					if v.Args[0] != fun.values[arg0] || v.Args[1] != fun.values[arg1] {
						t.Fatalf("phi arguments changed: %s", v.LongString())
					}
				} else if len(v.Args) != 1 || v.Args[0] != fun.values["cond"] {
					t.Fatalf("phi should use entry condition: %s", v.LongString())
				}
			})
		}
	}
}
