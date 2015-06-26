// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"testing"
)

func BenchmarkDominatorsLinear(b *testing.B)     { benchmarkDominators(b, 10000, genLinear) }
func BenchmarkDominatorsFwdBack(b *testing.B)    { benchmarkDominators(b, 10000, genFwdBack) }
func BenchmarkDominatorsManyPred(b *testing.B)   { benchmarkDominators(b, 10000, genManyPred) }
func BenchmarkDominatorsMaxPred(b *testing.B)    { benchmarkDominators(b, 10000, genMaxPred) }
func BenchmarkDominatorsMaxPredVal(b *testing.B) { benchmarkDominators(b, 10000, genMaxPredValue) }

type blockGen func(size int) []bloc

// genLinear creates an array of blocks that succeed one another
// b_n -> [b_n+1].
func genLinear(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Goto(blockn(0)),
		),
	)
	for i := 0; i < size; i++ {
		blocs = append(blocs, Bloc(blockn(i),
			Goto(blockn(i+1))))
	}

	blocs = append(blocs,
		Bloc(blockn(size), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	return blocs
}

// genLinear creates an array of blocks that alternate between
// b_n -> [b_n+1], b_n -> [b_n+1, b_n-1] , b_n -> [b_n+1, b_n+2]
func genFwdBack(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			Goto(blockn(0)),
		),
	)
	for i := 0; i < size; i++ {
		switch i % 2 {
		case 0:
			blocs = append(blocs, Bloc(blockn(i),
				If("p", blockn(i+1), blockn(i+2))))
		case 1:
			blocs = append(blocs, Bloc(blockn(i),
				If("p", blockn(i+1), blockn(i-1))))
		}
	}

	blocs = append(blocs,
		Bloc(blockn(size), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	return blocs
}

// genManyPred creates an array of blocks where 1/3rd have a sucessor of the
// first block, 1/3rd the last block, and the remaining third are plain.
func genManyPred(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			Goto(blockn(0)),
		),
	)

	// We want predecessor lists to be long, so 2/3rds of the blocks have a
	// sucessor of the first or last block.
	for i := 0; i < size; i++ {
		switch i % 3 {
		case 0:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConst, TypeBool, 0, true),
				Goto(blockn(i+1))))
		case 1:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConst, TypeBool, 0, true),
				If("p", blockn(i+1), blockn(0))))
		case 2:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConst, TypeBool, 0, true),
				If("p", blockn(i+1), blockn(size))))
		}
	}

	blocs = append(blocs,
		Bloc(blockn(size), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	return blocs
}

// genMaxPred maximizes the size of the 'exit' predecessor list.
func genMaxPred(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			Goto(blockn(0)),
		),
	)

	for i := 0; i < size; i++ {
		blocs = append(blocs, Bloc(blockn(i),
			If("p", blockn(i+1), "exit")))
	}

	blocs = append(blocs,
		Bloc(blockn(size), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	return blocs
}

// genMaxPredValue is identical to genMaxPred but contains an
// additional value.
func genMaxPredValue(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			Goto(blockn(0)),
		),
	)

	for i := 0; i < size; i++ {
		blocs = append(blocs, Bloc(blockn(i),
			Valu("a", OpConst, TypeBool, 0, true),
			If("p", blockn(i+1), "exit")))
	}

	blocs = append(blocs,
		Bloc(blockn(size), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	return blocs
}

// sink for benchmark
var domBenchRes []*Block

func benchmarkDominators(b *testing.B, size int, bg blockGen) {
	c := NewConfig("amd64", DummyFrontend{b})
	fun := Fun(c, "entry", bg(size)...)

	CheckFunc(fun.f)
	b.SetBytes(int64(size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		domBenchRes = dominators(fun.f)
	}
}

func verifyDominators(t *testing.T, f fun, doms map[string]string) {
	blockNames := map[*Block]string{}
	for n, b := range f.blocks {
		blockNames[b] = n
	}

	calcDom := dominators(f.f)

	for n, d := range doms {
		nblk, ok := f.blocks[n]
		if !ok {
			t.Errorf("invalid block name %s", n)
		}
		dblk, ok := f.blocks[d]
		if !ok {
			t.Errorf("invalid block name %s", d)
		}

		domNode := calcDom[nblk.ID]
		switch {
		case calcDom[nblk.ID] == dblk:
			calcDom[nblk.ID] = nil
			continue
		case calcDom[nblk.ID] != dblk:
			t.Errorf("expected %s as dominator of %s, found %s", d, n, blockNames[domNode])
		default:
			t.Fatal("unexpected dominator condition")
		}
	}

	for id, d := range calcDom {
		// If nil, we've already verified it
		if d == nil {
			continue
		}
		for _, b := range f.blocks {
			if int(b.ID) == id {
				t.Errorf("unexpected dominator of %s for %s", blockNames[d], blockNames[b])
			}
		}
	}

}

func TestDominatorsSimple(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Goto("a")),
		Bloc("a",
			Goto("b")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	doms := map[string]string{
		"a":    "entry",
		"b":    "a",
		"c":    "b",
		"exit": "c",
	}

	verifyDominators(t, fun, doms)

}

func TestDominatorsMultPredFwd(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			If("p", "a", "c")),
		Bloc("a",
			If("p", "b", "c")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	doms := map[string]string{
		"a":    "entry",
		"b":    "a",
		"c":    "entry",
		"exit": "c",
	}

	verifyDominators(t, fun, doms)

}

func TestDominatorsMultPredRev(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			Goto("a")),
		Bloc("a",
			If("p", "b", "entry")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			If("p", "exit", "b")),
		Bloc("exit",
			Exit("mem")))

	doms := map[string]string{
		"a":    "entry",
		"b":    "a",
		"c":    "b",
		"exit": "c",
	}
	verifyDominators(t, fun, doms)
}

func TestDominatorsMultPred(t *testing.T) {
	c := NewConfig("amd64", DummyFrontend{t})
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpArg, TypeMem, 0, ".mem"),
			Valu("p", OpConst, TypeBool, 0, true),
			If("p", "a", "c")),
		Bloc("a",
			If("p", "b", "c")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			If("p", "b", "exit")),
		Bloc("exit",
			Exit("mem")))

	doms := map[string]string{
		"a":    "entry",
		"b":    "entry",
		"c":    "entry",
		"exit": "c",
	}
	verifyDominators(t, fun, doms)
}
