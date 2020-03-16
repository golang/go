// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ssa

import (
	"cmd/compile/internal/types"
	"fmt"
	"testing"
)

const (
	blockCount = 1000
	passCount  = 15000
)

type passFunc func(*Func)

func BenchmarkDSEPass(b *testing.B)           { benchFnPass(b, dse, blockCount, genFunction) }
func BenchmarkDSEPassBlock(b *testing.B)      { benchFnBlock(b, dse, genFunction) }
func BenchmarkCSEPass(b *testing.B)           { benchFnPass(b, cse, blockCount, genFunction) }
func BenchmarkCSEPassBlock(b *testing.B)      { benchFnBlock(b, cse, genFunction) }
func BenchmarkDeadcodePass(b *testing.B)      { benchFnPass(b, deadcode, blockCount, genFunction) }
func BenchmarkDeadcodePassBlock(b *testing.B) { benchFnBlock(b, deadcode, genFunction) }

func multi(f *Func) {
	cse(f)
	dse(f)
	deadcode(f)
}
func BenchmarkMultiPass(b *testing.B)      { benchFnPass(b, multi, blockCount, genFunction) }
func BenchmarkMultiPassBlock(b *testing.B) { benchFnBlock(b, multi, genFunction) }

// benchFnPass runs passFunc b.N times across a single function.
func benchFnPass(b *testing.B, fn passFunc, size int, bg blockGen) {
	b.ReportAllocs()
	c := testConfig(b)
	fun := c.Fun("entry", bg(size)...)
	CheckFunc(fun.f)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fn(fun.f)
		b.StopTimer()
		CheckFunc(fun.f)
		b.StartTimer()
	}
}

// benchFnPass runs passFunc across a function with b.N blocks.
func benchFnBlock(b *testing.B, fn passFunc, bg blockGen) {
	b.ReportAllocs()
	c := testConfig(b)
	fun := c.Fun("entry", bg(b.N)...)
	CheckFunc(fun.f)
	b.ResetTimer()
	for i := 0; i < passCount; i++ {
		fn(fun.f)
	}
	b.StopTimer()
}

func genFunction(size int) []bloc {
	var blocs []bloc
	elemType := types.Types[types.TINT64]
	ptrType := elemType.PtrTo()

	valn := func(s string, m, n int) string { return fmt.Sprintf("%s%d-%d", s, m, n) }
	blocs = append(blocs,
		Bloc("entry",
			Valu(valn("store", 0, 4), OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, types.Types[types.TUINTPTR], 0, nil),
			Goto(blockn(1)),
		),
	)
	for i := 1; i < size+1; i++ {
		blocs = append(blocs, Bloc(blockn(i),
			Valu(valn("v", i, 0), OpConstBool, types.Types[types.TBOOL], 1, nil),
			Valu(valn("addr", i, 1), OpAddr, ptrType, 0, nil, "sb"),
			Valu(valn("addr", i, 2), OpAddr, ptrType, 0, nil, "sb"),
			Valu(valn("addr", i, 3), OpAddr, ptrType, 0, nil, "sb"),
			Valu(valn("zero", i, 1), OpZero, types.TypeMem, 8, elemType, valn("addr", i, 3),
				valn("store", i-1, 4)),
			Valu(valn("store", i, 1), OpStore, types.TypeMem, 0, elemType, valn("addr", i, 1),
				valn("v", i, 0), valn("zero", i, 1)),
			Valu(valn("store", i, 2), OpStore, types.TypeMem, 0, elemType, valn("addr", i, 2),
				valn("v", i, 0), valn("store", i, 1)),
			Valu(valn("store", i, 3), OpStore, types.TypeMem, 0, elemType, valn("addr", i, 1),
				valn("v", i, 0), valn("store", i, 2)),
			Valu(valn("store", i, 4), OpStore, types.TypeMem, 0, elemType, valn("addr", i, 3),
				valn("v", i, 0), valn("store", i, 3)),
			Goto(blockn(i+1))))
	}

	blocs = append(blocs,
		Bloc(blockn(size+1), Goto("exit")),
		Bloc("exit", Exit("store0-4")),
	)

	return blocs
}
