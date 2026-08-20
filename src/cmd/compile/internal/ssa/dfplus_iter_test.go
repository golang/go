// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"iter"
	"slices"
	"testing"

	"cmd/compile/internal/ssa/block"
)

// genCrossLadder builds a k-column, 2-rail cross ladder CFG:
//
//	entry → L1  R1
//	        │╲ ╱│     every block branches to BOTH blocks of
//	        │ ╳ │     the next column
//	        │╱ ╲│
//	        L2  R2
//	         ...
//	        Lk  Rk
//	         ↘  ↙
//	         exit
//
// The cross edges help to do quadratic DF+ walks (assume each of N blocks has another def
// and needs to iterate with f.IterDomFrontierPlus O(N) blocks).
// Returned are the function and its column blocks, column by column.
func genCrossLadder(k int) (*Func, []*Block) {
	f := (&Config{}).NewFunc(nil, &Cache{})
	entry := f.NewBlock(block.BlockIf)
	f.Entry = entry
	exit := f.NewBlock(block.BlockExit)
	col := make([]*Block, 0, 2*k)
	prevL, prevR := entry, entry
	for i := 0; i < k; i++ {
		kind := block.BlockIf
		if i == k-1 {
			kind = block.BlockPlain // goto exit
		}
		l := f.NewBlock(kind)
		r := f.NewBlock(kind)
		col = append(col, l, r)
		prevL.AddEdgeTo(l)
		prevL.AddEdgeTo(r)
		prevR.AddEdgeTo(l)
		prevR.AddEdgeTo(r)
		prevL, prevR = l, r
	}
	col[len(col)-2].AddEdgeTo(exit)
	col[len(col)-1].AddEdgeTo(exit)
	return f, col
}

// BenchmarkIterDomFrontierPlus walks the iterated dominance frontier of every
// column block of a cross ladder, the merge-set stress shape.
// Ideally, ns/block (ns/op ÷ blocks/op) stays flat as k grows.
func BenchmarkIterDomFrontierPlus(b *testing.B) {
	for _, k := range []int{8, 16, 32} {
		b.Run(fmt.Sprintf("k=%d", k), func(b *testing.B) {
			f, col := genCrossLadder(k)
			b.ReportAllocs()
			b.ResetTimer()
			var n int
			for i := 0; i < b.N; i++ {
				for j := range col {
					for range f.IterDomFrontierPlus(slices.Values(col[j : j+1])) {
						n++
					}
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(n)/float64(b.N), "blocks/op")
			if n != 2*k*k*b.N {
				b.Fatalf("walked %d blocks per round, want %d", n/b.N, 2*k*k)
			}
		})
	}
}

// TestIterDomFrontierPlusSeedAtMerge tests DF+ of a small CFG with a loop
// where its header is both a seed and a merge point.
// A def in an unreachable block (u → b3) must not join the merge set,
// and the header (b2) must be in it.
func TestIterDomFrontierPlusSeedAtMerge(t *testing.T) {
	//
	//	x := 0
	// loop:
	//	x = x + 1         // b2: def in the loop head
	//	if c { continue } // b4 → b2, back edge 1
	//	if d { break }    // b5 → b3
	//	goto loop         // b5 → b2, back edge 2
	//	return x          // b3
	// unreachable: x = 9; // u → b3
	//
	f := (&Config{}).NewFunc(nil, &Cache{})
	b1 := f.NewBlock(block.BlockPlain) // entry: x:=0
	f.Entry = b1
	b2 := f.NewBlock(block.BlockIf)    // loop head: x = x + 1; if c
	b4 := f.NewBlock(block.BlockPlain) // { continue }
	b5 := f.NewBlock(block.BlockIf)    // if d { break }; goto loop
	b3 := f.NewBlock(block.BlockExit)
	u := f.NewBlock(block.BlockPlain) // unreachable; goto b3

	b1.AddEdgeTo(b2)
	b2.AddEdgeTo(b4)
	b2.AddEdgeTo(b5)
	b4.AddEdgeTo(b2) // back edge 1
	b5.AddEdgeTo(b3) // break
	b5.AddEdgeTo(b2) // back edge 2
	u.AddEdgeTo(b3)  // goto b3

	got := collectBlockIDs(f.IterDomFrontierPlus(slices.Values([]*Block{b1, b2, u})))
	if want := []ID{b2.ID}; !slices.Equal(got, want) {
		t.Errorf("got DF+ = %v, want %v", got, want)
	}
}

func collectBlockIDs(seq iter.Seq[*Block]) []ID {
	var ids []ID
	for b := range seq {
		ids = append(ids, b.ID)
	}
	return ids
}
