// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
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
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
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

// genFwdBack creates an array of blocks that alternate between
// b_n -> [b_n+1], b_n -> [b_n+1, b_n-1] , b_n -> [b_n+1, b_n+2]
func genFwdBack(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
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

// genManyPred creates an array of blocks where 1/3rd have a successor of the
// first block, 1/3rd the last block, and the remaining third are plain.
func genManyPred(size int) []bloc {
	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto(blockn(0)),
		),
	)

	// We want predecessor lists to be long, so 2/3rds of the blocks have a
	// successor of the first or last block.
	for i := 0; i < size; i++ {
		switch i % 3 {
		case 0:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConstBool, types.Types[types.TBOOL], 1, nil),
				Goto(blockn(i+1))))
		case 1:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConstBool, types.Types[types.TBOOL], 1, nil),
				If("p", blockn(i+1), blockn(0))))
		case 2:
			blocs = append(blocs, Bloc(blockn(i),
				Valu("a", OpConstBool, types.Types[types.TBOOL], 1, nil),
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
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
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
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto(blockn(0)),
		),
	)

	for i := 0; i < size; i++ {
		blocs = append(blocs, Bloc(blockn(i),
			Valu("a", OpConstBool, types.Types[types.TBOOL], 1, nil),
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
	c := testConfig(b)
	fun := c.Fun("entry", bg(size)...)

	CheckFunc(fun.f)
	b.SetBytes(int64(size))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		domBenchRes = dominators(fun.f)
	}
}

type domFunc func(f *Func) []*Block

// verifyDominators verifies that the dominators of fut (function under test)
// as determined by domFn, match the map node->dominator
func verifyDominators(t *testing.T, fut fun, domFn domFunc, doms map[string]string) {
	blockNames := map[*Block]string{}
	for n, b := range fut.blocks {
		blockNames[b] = n
	}

	calcDom := domFn(fut.f)

	for n, d := range doms {
		nblk, ok := fut.blocks[n]
		if !ok {
			t.Errorf("invalid block name %s", n)
		}
		dblk, ok := fut.blocks[d]
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
		for _, b := range fut.blocks {
			if int(b.ID) == id {
				t.Errorf("unexpected dominator of %s for %s", blockNames[d], blockNames[b])
			}
		}
	}

}

func TestDominatorsSingleBlock(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Exit("mem")))

	doms := map[string]string{}

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)

}

func TestDominatorsSimple(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
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

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)

}

func TestDominatorsMultPredFwd(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
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

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)
}

func TestDominatorsDeadCode(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 0, nil),
			If("p", "b3", "b5")),
		Bloc("b2", Exit("mem")),
		Bloc("b3", Goto("b2")),
		Bloc("b4", Goto("b2")),
		Bloc("b5", Goto("b2")))

	doms := map[string]string{
		"b2": "entry",
		"b3": "entry",
		"b5": "entry",
	}

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)
}

func TestDominatorsMultPredRev(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Goto("first")),
		Bloc("first",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("a")),
		Bloc("a",
			If("p", "b", "first")),
		Bloc("b",
			Goto("c")),
		Bloc("c",
			If("p", "exit", "b")),
		Bloc("exit",
			Exit("mem")))

	doms := map[string]string{
		"first": "entry",
		"a":     "first",
		"b":     "a",
		"c":     "b",
		"exit":  "c",
	}

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)
}

func TestDominatorsMultPred(t *testing.T) {
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
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

	CheckFunc(fun.f)
	verifyDominators(t, fun, dominators, doms)
	verifyDominators(t, fun, dominatorsSimple, doms)
}

func TestInfiniteLoop(t *testing.T) {
	c := testConfig(t)
	// note lack of an exit block
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			Goto("a")),
		Bloc("a",
			Goto("b")),
		Bloc("b",
			Goto("a")))

	CheckFunc(fun.f)
	doms := map[string]string{"a": "entry",
		"b": "a"}
	verifyDominators(t, fun, dominators, doms)
}

func TestDomTricky(t *testing.T) {
	doms := map[string]string{
		"4":  "1",
		"2":  "4",
		"5":  "4",
		"11": "4",
		"15": "4", // the incorrect answer is "5"
		"10": "15",
		"19": "15",
	}

	if4 := [2]string{"2", "5"}
	if5 := [2]string{"15", "11"}
	if15 := [2]string{"19", "10"}

	for i := 0; i < 8; i++ {
		a := 1 & i
		b := 1 & i >> 1
		c := 1 & i >> 2

		cfg := testConfig(t)
		fun := cfg.Fun("1",
			Bloc("1",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
				Goto("4")),
			Bloc("2",
				Goto("11")),
			Bloc("4",
				If("p", if4[a], if4[1-a])), // 2, 5
			Bloc("5",
				If("p", if5[b], if5[1-b])), //15, 11
			Bloc("10",
				Exit("mem")),
			Bloc("11",
				Goto("15")),
			Bloc("15",
				If("p", if15[c], if15[1-c])), //19, 10
			Bloc("19",
				Goto("10")))
		CheckFunc(fun.f)
		verifyDominators(t, fun, dominators, doms)
		verifyDominators(t, fun, dominatorsSimple, doms)
	}
}

// generateDominatorMap uses dominatorsSimple to obtain a
// reference dominator tree for testing faster algorithms.
func generateDominatorMap(fut fun) map[string]string {
	blockNames := map[*Block]string{}
	for n, b := range fut.blocks {
		blockNames[b] = n
	}
	referenceDom := dominatorsSimple(fut.f)
	doms := make(map[string]string)
	for _, b := range fut.f.Blocks {
		if d := referenceDom[b.ID]; d != nil {
			doms[blockNames[b]] = blockNames[d]
		}
	}
	return doms
}

func TestDominatorsPostTrickyA(t *testing.T) {
	testDominatorsPostTricky(t, "b8", "b11", "b10", "b8", "b14", "b15")
}

func TestDominatorsPostTrickyB(t *testing.T) {
	testDominatorsPostTricky(t, "b11", "b8", "b10", "b8", "b14", "b15")
}

func TestDominatorsPostTrickyC(t *testing.T) {
	testDominatorsPostTricky(t, "b8", "b11", "b8", "b10", "b14", "b15")
}

func TestDominatorsPostTrickyD(t *testing.T) {
	testDominatorsPostTricky(t, "b11", "b8", "b8", "b10", "b14", "b15")
}

func TestDominatorsPostTrickyE(t *testing.T) {
	testDominatorsPostTricky(t, "b8", "b11", "b10", "b8", "b15", "b14")
}

func TestDominatorsPostTrickyF(t *testing.T) {
	testDominatorsPostTricky(t, "b11", "b8", "b10", "b8", "b15", "b14")
}

func TestDominatorsPostTrickyG(t *testing.T) {
	testDominatorsPostTricky(t, "b8", "b11", "b8", "b10", "b15", "b14")
}

func TestDominatorsPostTrickyH(t *testing.T) {
	testDominatorsPostTricky(t, "b11", "b8", "b8", "b10", "b15", "b14")
}

func testDominatorsPostTricky(t *testing.T, b7then, b7else, b12then, b12else, b13then, b13else string) {
	c := testConfig(t)
	fun := c.Fun("b1",
		Bloc("b1",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpConstBool, types.Types[types.TBOOL], 1, nil),
			If("p", "b3", "b2")),
		Bloc("b3",
			If("p", "b5", "b6")),
		Bloc("b5",
			Goto("b7")),
		Bloc("b7",
			If("p", b7then, b7else)),
		Bloc("b8",
			Goto("b13")),
		Bloc("b13",
			If("p", b13then, b13else)),
		Bloc("b14",
			Goto("b10")),
		Bloc("b15",
			Goto("b16")),
		Bloc("b16",
			Goto("b9")),
		Bloc("b9",
			Goto("b7")),
		Bloc("b11",
			Goto("b12")),
		Bloc("b12",
			If("p", b12then, b12else)),
		Bloc("b10",
			Goto("b6")),
		Bloc("b6",
			Goto("b17")),
		Bloc("b17",
			Goto("b18")),
		Bloc("b18",
			If("p", "b22", "b19")),
		Bloc("b22",
			Goto("b23")),
		Bloc("b23",
			If("p", "b21", "b19")),
		Bloc("b19",
			If("p", "b24", "b25")),
		Bloc("b24",
			Goto("b26")),
		Bloc("b26",
			Goto("b25")),
		Bloc("b25",
			If("p", "b27", "b29")),
		Bloc("b27",
			Goto("b30")),
		Bloc("b30",
			Goto("b28")),
		Bloc("b29",
			Goto("b31")),
		Bloc("b31",
			Goto("b28")),
		Bloc("b28",
			If("p", "b32", "b33")),
		Bloc("b32",
			Goto("b21")),
		Bloc("b21",
			Goto("b47")),
		Bloc("b47",
			If("p", "b45", "b46")),
		Bloc("b45",
			Goto("b48")),
		Bloc("b48",
			Goto("b49")),
		Bloc("b49",
			If("p", "b50", "b51")),
		Bloc("b50",
			Goto("b52")),
		Bloc("b52",
			Goto("b53")),
		Bloc("b53",
			Goto("b51")),
		Bloc("b51",
			Goto("b54")),
		Bloc("b54",
			Goto("b46")),
		Bloc("b46",
			Exit("mem")),
		Bloc("b33",
			Goto("b34")),
		Bloc("b34",
			Goto("b37")),
		Bloc("b37",
			If("p", "b35", "b36")),
		Bloc("b35",
			Goto("b38")),
		Bloc("b38",
			Goto("b39")),
		Bloc("b39",
			If("p", "b40", "b41")),
		Bloc("b40",
			Goto("b42")),
		Bloc("b42",
			Goto("b43")),
		Bloc("b43",
			Goto("b41")),
		Bloc("b41",
			Goto("b44")),
		Bloc("b44",
			Goto("b36")),
		Bloc("b36",
			Goto("b20")),
		Bloc("b20",
			Goto("b18")),
		Bloc("b2",
			Goto("b4")),
		Bloc("b4",
			Exit("mem")))
	CheckFunc(fun.f)
	doms := generateDominatorMap(fun)
	verifyDominators(t, fun, dominators, doms)
}
