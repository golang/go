// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"iter"
	"runtime"
	"slices"
	"sync/atomic"
	"testing"
	"testing/synctest"
)

const rangedTwice = "logos: group stage sequence ranged twice"

func TestGroupStageSecondRangePanicsOnEveryEntry(t *testing.T) {
	eachEntry(t, func(t *testing.T, run entryFunc[int]) {
		seq := run(t.Context(), 2, 3, func(_ context.Context, i int) (int, bool, error) { return i, true, nil })
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != 3 {
			t.Fatalf("first range: rs = %v, err = %v, want 3 results and nil", d.rs, d.err)
		}
		got := panicOf(func() {
			for range seq {
			}
		})
		if got != rangedTwice {
			t.Fatalf("second range panic = %v, want %q", got, rangedTwice)
		}
	})
}

func TestGroupStageDoesNothingUntilRangedOnSequenceEntries(t *testing.T) {
	for name, entry := range map[string]func(context.Context, iter.Seq[int], func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]{
		"seq": func(ctx context.Context, src iter.Seq[int], f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupSeq(ctx, 2, src, f)
		},
		"seq-unordered": func(ctx context.Context, src iter.Seq[int], f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupSeqUnordered(ctx, 2, src, f)
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var pulled, calls atomic.Int32
				src := func(yield func(int) bool) {
					pulled.Add(1)
					yield(0)
				}
				base := runtime.NumGoroutine()
				seq := entry(t.Context(), src, func(_ context.Context, i int) (int, bool, error) {
					calls.Add(1)
					return i, true, nil
				})
				synctest.Wait()
				if got := goroutinesAtMost(base); pulled.Load() != 0 || calls.Load() != 0 || got > base {
					t.Fatalf("before the range: pulled = %d, calls = %d, goroutines %d -> %d", pulled.Load(), calls.Load(), base, got)
				}
				d := finished(t, startStage(seq, nil))
				if d.err != nil || !slices.Equal(d.rs, []int{0}) {
					t.Fatalf("rs = %v, err = %v, want [0], nil", d.rs, d.err)
				}
			})
		})
	}
}

func TestGroupStageDoesNothingUntilRangedOnChannelEntries(t *testing.T) {
	for name, entry := range map[string]func(context.Context, chan int, func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]{
		"chan": func(ctx context.Context, src chan int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupChan(ctx, 2, src, f)
		},
		"chan-unordered": func(ctx context.Context, src chan int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupChanUnordered(ctx, 2, src, f)
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var calls atomic.Int32
				ch := make(chan int, 1)
				ch <- 7
				base := runtime.NumGoroutine()
				seq := entry(t.Context(), ch, func(_ context.Context, i int) (int, bool, error) {
					calls.Add(1)
					return i, true, nil
				})
				synctest.Wait()
				if got := goroutinesAtMost(base); len(ch) != 1 || calls.Load() != 0 || got > base {
					t.Fatalf("before the range: queued = %d, calls = %d, goroutines %d -> %d", len(ch), calls.Load(), base, got)
				}
				close(ch)
				d := finished(t, startStage(seq, nil))
				if d.err != nil || !slices.Equal(d.rs, []int{7}) {
					t.Fatalf("rs = %v, err = %v, want [7], nil", d.rs, d.err)
				}
			})
		})
	}
}

func TestGroupStageIgnoresSourceElementsAfterTheStop(t *testing.T) {
	for name, entry := range map[string]func(context.Context, iter.Seq[int], func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]{
		"seq": func(ctx context.Context, src iter.Seq[int], f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupSeq(ctx, 1, src, f)
		},
		"seq-unordered": func(ctx context.Context, src iter.Seq[int], f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupSeqUnordered(ctx, 1, src, f)
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var calls, yields atomic.Int32
				rude := func(yield func(int) bool) {
					for i := range 8 {
						yields.Add(1)
						yield(i)
					}
				}
				seq := entry(t.Context(), rude, func(_ context.Context, i int) (int, bool, error) {
					calls.Add(1)
					return i, true, nil
				})
				d := finished(t, startStage(seq, func(int) bool { return false }))
				if d.err != nil || d.pan != nil || !slices.Equal(d.rs, []int{0}) {
					t.Fatalf("rs = %v, err = %v, panic = %v, want [0], nil, nil", d.rs, d.err, d.pan)
				}
				if yields.Load() != 8 || calls.Load() != 1 {
					t.Fatalf("source yields = %d, calls of f = %d, want 8 and 1: no child may start for an element after the stop", yields.Load(), calls.Load())
				}
			})
		})
	}
}

func TestGroupChanStopsReceivingAfterTheStop(t *testing.T) {
	for name, entry := range map[string]func(context.Context, <-chan int, func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]{
		"chan": func(ctx context.Context, src <-chan int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupChan(ctx, 1, src, f)
		},
		"chan-unordered": func(ctx context.Context, src <-chan int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
			return GroupChanUnordered(ctx, 1, src, f)
		},
	} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				ch := make(chan int, 8)
				for i := range 8 {
					ch <- i
				}
				close(ch)
				var calls atomic.Int32
				seq := entry(t.Context(), ch, func(_ context.Context, i int) (int, bool, error) {
					calls.Add(1)
					return i, true, nil
				})
				d := finished(t, startStage(seq, func(int) bool { return false }))
				if d.err != nil || d.pan != nil || !slices.Equal(d.rs, []int{0}) {
					t.Fatalf("rs = %v, err = %v, panic = %v, want [0], nil, nil", d.rs, d.err, d.pan)
				}
				if calls.Load() != 1 || len(ch) != 7 {
					t.Fatalf("calls of f = %d, elements left = %d, want 1 and 7: the stage may not pull or start a child after the stop", calls.Load(), len(ch))
				}
			})
		})
	}
}
