package ssa

import (
	"strconv"
	"testing"
)

func BenchmarkNilCheckDeep1(b *testing.B)     { benchmarkNilCheckDeep(b, 1) }
func BenchmarkNilCheckDeep10(b *testing.B)    { benchmarkNilCheckDeep(b, 10) }
func BenchmarkNilCheckDeep100(b *testing.B)   { benchmarkNilCheckDeep(b, 100) }
func BenchmarkNilCheckDeep1000(b *testing.B)  { benchmarkNilCheckDeep(b, 1000) }
func BenchmarkNilCheckDeep10000(b *testing.B) { benchmarkNilCheckDeep(b, 10000) }

// benchmarkNilCheckDeep is a stress test of nilcheckelim.
// It uses the worst possible input: A linear string of
// nil checks, none of which can be eliminated.
// Run with multiple depths to observe big-O behavior.
func benchmarkNilCheckDeep(b *testing.B, depth int) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing

	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto(blockn(0)),
		),
	)
	for i := 0; i < depth; i++ {
		blocs = append(blocs,
			Bloc(blockn(i),
				Valu(ptrn(i), OpAddr, ptrType, 0, nil, "sb"),
				Valu(booln(i), OpIsNonNil, TypeBool, 0, nil, ptrn(i)),
				If(booln(i), blockn(i+1), "exit"),
			),
		)
	}
	blocs = append(blocs,
		Bloc(blockn(depth), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	c := NewConfig("amd64", DummyFrontend{b})
	fun := Fun(c, "entry", blocs...)

	CheckFunc(fun.f)
	b.SetBytes(int64(depth)) // helps for eyeballing linearity
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		nilcheckelim(fun.f)
	}
}

func blockn(n int) string { return "b" + strconv.Itoa(n) }
func ptrn(n int) string   { return "p" + strconv.Itoa(n) }
func booln(n int) string  { return "c" + strconv.Itoa(n) }

func isNilCheck(b *Block) bool {
	return b.Kind == BlockIf && b.Control.Op == OpIsNonNil
}

// TestNilcheckSimple verifies that a second repeated nilcheck is removed.
func TestNilcheckSimple(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpConstPtr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "secondCheck", "exit")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
	}
}

// TestNilcheckDomOrder ensures that the nil check elimination isn't dependant
// on the order of the dominees.
func TestNilcheckDomOrder(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpConstPtr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "secondCheck", "exit")),
		Bloc("exit",
			Exit("mem")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
	}
}

// TestNilcheckAddr verifies that nilchecks of OpAddr constructed values are removed.
func TestNilcheckAddr(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpAddr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["checkPtr"] && isNilCheck(b) {
			t.Errorf("checkPtr was not eliminated")
		}
	}
}

// TestNilcheckAddPtr verifies that nilchecks of OpAddPtr constructed values are removed.
func TestNilcheckAddPtr(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpAddPtr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["checkPtr"] && isNilCheck(b) {
			t.Errorf("checkPtr was not eliminated")
		}
	}
}

// TestNilcheckKeepRemove verifies that dupliate checks of the same pointer
// are removed, but checks of different pointers are not.
func TestNilcheckKeepRemove(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpConstPtr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "differentCheck", "exit")),
		Bloc("differentCheck",
			Valu("ptr2", OpConstPtr, ptrType, 0, nil, "sb"),
			Valu("bool2", OpIsNonNil, TypeBool, 0, nil, "ptr2"),
			If("bool2", "secondCheck", "exit")),
		Bloc("secondCheck",
			Valu("bool3", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool3", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	foundDifferentCheck := false
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
		if b == fun.blocks["differentCheck"] && isNilCheck(b) {
			foundDifferentCheck = true
		}
	}
	if !foundDifferentCheck {
		t.Errorf("removed differentCheck, but shouldn't have")
	}
}

// TestNilcheckInFalseBranch tests that nil checks in the false branch of an nilcheck
// block are *not* removed.
func TestNilcheckInFalseBranch(t *testing.T) {
	ptrType := &TypeImpl{Size_: 8, Ptr: true, Name: "testptr"} // dummy for testing
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("sb", OpSB, TypeInvalid, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpConstPtr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool1", "extra", "secondCheck")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool2", "extra", "thirdCheck")),
		Bloc("thirdCheck",
			Valu("bool3", OpIsNonNil, TypeBool, 0, nil, "ptr1"),
			If("bool3", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f)
	deadcode(fun.f)

	CheckFunc(fun.f)
	foundSecondCheck := false
	foundThirdCheck := false
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			foundSecondCheck = true
		}
		if b == fun.blocks["thirdCheck"] && isNilCheck(b) {
			foundThirdCheck = true
		}
	}
	if !foundSecondCheck {
		t.Errorf("removed secondCheck, but shouldn't have [false branch]")
	}
	if !foundThirdCheck {
		t.Errorf("removed thirdCheck, but shouldn't have [false branch]")
	}
}
