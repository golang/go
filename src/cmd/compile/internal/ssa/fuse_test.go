package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
	"strconv"
	"testing"
)

func TestFuseEliminatesOneBranch(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, c.config.Types.Bool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "exit")),
		Bloc("then",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	fuseLate(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
	}
}

func TestFuseEliminatesBothBranches(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, c.config.Types.Bool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	fuseLate(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
		if b == fun.blocks["else"] && b.Kind != BlockInvalid {
			t.Errorf("else was not eliminated, but should have")
		}
	}
}

func TestFuseHandlesPhis(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, c.config.Types.Bool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Valu("phi", OpPhi, ptrType, 0, nil, "ptr1", "ptr1"),
			Exit("mem")))

	CheckFunc(fun.f)
	fuseLate(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
		if b == fun.blocks["else"] && b.Kind != BlockInvalid {
			t.Errorf("else was not eliminated, but should have")
		}
	}
}

func TestFuseEliminatesEmptyBlocks(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("z0")),
		Bloc("z1",
			Goto("z2")),
		Bloc("z3",
			Goto("exit")),
		Bloc("z2",
			Goto("z3")),
		Bloc("z0",
			Goto("z1")),
		Bloc("exit",
			Exit("mem"),
		))

	CheckFunc(fun.f)
	fuseLate(fun.f)

	for k, b := range fun.blocks {
		if k[:1] == "z" && b.Kind != BlockInvalid {
			t.Errorf("%s was not eliminated, but should have", k)
		}
	}
}

func TestFuseSideEffects(t *testing.T) {
	// Test that we don't fuse branches that have side effects but
	// have no use (e.g. followed by infinite loop).
	// See issue #36005.
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("b", OpArg, c.config.Types.Bool, 0, nil),
			If("b", "then", "else")),
		Bloc("then",
			Valu("call1", OpStaticCall, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Goto("empty")),
		Bloc("else",
			Valu("call2", OpStaticCall, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Goto("empty")),
		Bloc("empty",
			Goto("loop")),
		Bloc("loop",
			Goto("loop")))

	CheckFunc(fun.f)
	fuseLate(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind == BlockInvalid {
			t.Errorf("then is eliminated, but should not")
		}
		if b == fun.blocks["else"] && b.Kind == BlockInvalid {
			t.Errorf("else is eliminated, but should not")
		}
	}
}

func BenchmarkFuse(b *testing.B) {
	for _, n := range [...]int{1, 10, 100, 1000, 10000} {
		b.Run(strconv.Itoa(n), func(b *testing.B) {
			c := testConfig(b)

			blocks := make([]bloc, 0, 2*n+3)
			blocks = append(blocks,
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("cond", OpArg, c.config.Types.Bool, 0, nil),
					Valu("x", OpArg, c.config.Types.Int64, 0, nil),
					Goto("exit")))

			phiArgs := make([]string, 0, 2*n)
			for i := 0; i < n; i++ {
				cname := fmt.Sprintf("c%d", i)
				blocks = append(blocks,
					Bloc(fmt.Sprintf("b%d", i), If("cond", cname, "merge")),
					Bloc(cname, Goto("merge")))
				phiArgs = append(phiArgs, "x", "x")
			}
			blocks = append(blocks,
				Bloc("merge",
					Valu("phi", OpPhi, types.TypeMem, 0, nil, phiArgs...),
					Goto("exit")),
				Bloc("exit",
					Exit("mem")))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fun := c.Fun("entry", blocks...)
				fuseLate(fun.f)
			}
		})
	}
}
