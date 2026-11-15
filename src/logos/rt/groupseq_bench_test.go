// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"iter"
	"strconv"
	"testing"
)

const benchElements = 10000

type benchShape struct {
	name string
	work func(b *testing.B) func(int) int
}

var benchShapes = []benchShape{
	{"cpu", cpuWork},
	{"block", roundTripWork},
}

func cpuWork(*testing.B) func(int) int {
	return func(i int) int {
		x := uint32(i)
		for range 64 {
			x = x*1664525 + 1013904223
		}
		return int(x >> 16)
	}
}

type roundTrip struct {
	v     int
	reply chan int
}

func roundTripWork(b *testing.B) func(int) int {
	req := make(chan roundTrip)
	for range 8 {
		go func() {
			for r := range req {
				r.reply <- r.v*3 + 1
			}
		}()
	}
	b.Cleanup(func() { close(req) })
	return func(i int) int {
		reply := make(chan int)
		req <- roundTrip{i, reply}
		return <-reply
	}
}

func benchStage(b *testing.B, build func(b *testing.B, limit int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]) {
	for _, shape := range benchShapes {
		for _, limit := range []int{1, 8, 64} {
			b.Run(shape.name+"/limit="+strconv.Itoa(limit), func(b *testing.B) {
				work := shape.work(b)
				f := func(_ context.Context, i int) (int, bool, error) { return work(i), true, nil }
				want := 0
				for i := range benchElements {
					want += work(i)
				}
				b.ReportAllocs()
				b.ResetTimer()
				for range b.N {
					got := 0
					for r, err := range build(b, limit, f) {
						if err != nil {
							b.Fatal(err)
						}
						got += r
					}
					if got != want {
						b.Fatalf("sum = %d, want %d", got, want)
					}
				}
			})
		}
	}
}

func BenchmarkGroupSeq(b *testing.B) {
	benchStage(b, func(b *testing.B, limit int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
		return GroupSeq(context.Background(), limit, intSeq(benchElements), f)
	})
}

func BenchmarkGroupSeqUnordered(b *testing.B) {
	benchStage(b, func(b *testing.B, limit int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
		return GroupSeqUnordered(context.Background(), limit, intSeq(benchElements), f)
	})
}

func BenchmarkGroupChan(b *testing.B) {
	benchStage(b, func(b *testing.B, limit int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
		b.StopTimer()
		ch := intChan(benchElements)
		b.StartTimer()
		return GroupChan(context.Background(), limit, ch, f)
	})
}
