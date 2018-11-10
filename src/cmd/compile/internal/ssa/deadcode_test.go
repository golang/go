// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
	"strconv"
	"testing"
)

func TestDeadLoop(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")),
		// dead loop
		Bloc("deadblock",
			// dead value in dead block
			Valu("deadval", OpConstBool, c.config.Types.Bool, 1, nil),
			If("deadval", "deadblock", "exit")))

	CheckFunc(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["deadblock"] {
			t.Errorf("dead block not removed")
		}
		for _, v := range b.Values {
			if v == fun.values["deadval"] {
				t.Errorf("control value of dead block not removed")
			}
		}
	}
}

func TestDeadValue(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("deadval", OpConst64, c.config.Types.Int64, 37, nil),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	for _, b := range fun.f.Blocks {
		for _, v := range b.Values {
			if v == fun.values["deadval"] {
				t.Errorf("dead value not removed")
			}
		}
	}
}

func TestNeverTaken(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("cond", OpConstBool, c.config.Types.Bool, 0, nil),
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			If("cond", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	Opt(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)

	if fun.blocks["entry"].Kind != BlockPlain {
		t.Errorf("if(false) not simplified")
	}
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] {
			t.Errorf("then block still present")
		}
		for _, v := range b.Values {
			if v == fun.values["cond"] {
				t.Errorf("constant condition still present")
			}
		}
	}

}

func TestNestedDeadBlocks(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("cond", OpConstBool, c.config.Types.Bool, 0, nil),
			If("cond", "b2", "b4")),
		Bloc("b2",
			If("cond", "b3", "b4")),
		Bloc("b3",
			If("cond", "b3", "b4")),
		Bloc("b4",
			If("cond", "b3", "exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	Opt(fun.f)
	CheckFunc(fun.f)
	Deadcode(fun.f)
	CheckFunc(fun.f)
	if fun.blocks["entry"].Kind != BlockPlain {
		t.Errorf("if(false) not simplified")
	}
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["b2"] {
			t.Errorf("b2 block still present")
		}
		if b == fun.blocks["b3"] {
			t.Errorf("b3 block still present")
		}
		for _, v := range b.Values {
			if v == fun.values["cond"] {
				t.Errorf("constant condition still present")
			}
		}
	}
}

func BenchmarkDeadCode(b *testing.B) {
	for _, n := range [...]int{1, 10, 100, 1000, 10000, 100000, 200000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			c := testConfig(b)
			blocks := make([]bloc, 0, n+2)
			blocks = append(blocks,
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Goto("exit")))
			blocks = append(blocks, Bloc("exit", Exit("mem")))
			for i := 0; i < n; i++ {
				blocks = append(blocks, Bloc(fmt.Sprintf("dead%d", i), Goto("exit")))
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fun := c.Fun("entry", blocks...)
				Deadcode(fun.f)
			}
		})
	}
}
