package ssa

import (
	"fmt"
	"strconv"
	"testing"
)

func TestFuseEliminatesOneBranch(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t}, nil, true)
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, TypeBool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "exit")),
		Bloc("then",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	fuse(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
	}
}

func TestFuseEliminatesBothBranches(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t}, nil, true)
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, TypeBool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	fuse(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
		if b == fun.blocks["else"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
	}
}

func TestFuseHandlesPhis(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t}, nil, true)
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, TypeBool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "then", "else")),
		Bloc("then",
			Goto("exit")),
		Bloc("else",
			Goto("exit")),
		Bloc("exit",
			Valu("phi", OpPhi, ptrType, 0, nil, "ptr1", "ptr1"),
			Exit("mem")))

	CheckFunc(fun.f)
	fuse(fun.f)

	for _, b := range fun.f.Blocks {
		if b == fun.blocks["then"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
		if b == fun.blocks["else"] && b.Kind != BlockInvalid {
			t.Errorf("then was not eliminated, but should have")
		}
	}
}

func TestFuseEliminatesEmptyBlocks(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t}, nil, true)
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
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
	fuse(fun.f)

	for k, b := range fun.blocks {
		if k[:1] == "z" && b.Kind != BlockInvalid {
			t.Errorf("%s was not eliminated, but should have", k)
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
					Valu("mem", OpInitMem, TypeMem, 0, nil),
					Valu("cond", OpArg, TypeBool, 0, nil),
					Valu("x", OpArg, TypeInt64, 0, nil),
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
					Valu("phi", OpPhi, TypeMem, 0, nil, phiArgs...),
					Goto("exit")),
				Bloc("exit",
					Exit("mem")))

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				fun := Fun(c, "entry", blocks...)
				fuse(fun.f)
				fun.f.Free()
			}
		})
	}
}
