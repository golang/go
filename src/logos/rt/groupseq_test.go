// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"errors"
	"iter"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"
)

const stageDeadline = time.Minute

type stageFunc func(ctx context.Context, limit, n int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error]

func intSeq(n int) iter.Seq[int] {
	return func(yield func(int) bool) {
		for i := range n {
			if !yield(i) {
				return
			}
		}
	}
}

func intChan(n int) <-chan int {
	ch := make(chan int, n)
	for i := range n {
		ch <- i
	}
	close(ch)
	return ch
}

func seqStage(ctx context.Context, limit, n int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
	return GroupSeq(ctx, limit, intSeq(n), f)
}

func seqStageUnordered(ctx context.Context, limit, n int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
	return GroupSeqUnordered(ctx, limit, intSeq(n), f)
}

func chanStage(ctx context.Context, limit, n int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
	return GroupChan(ctx, limit, intChan(n), f)
}

func chanStageUnordered(ctx context.Context, limit, n int, f func(context.Context, int) (int, bool, error)) iter.Seq2[int, error] {
	return GroupChanUnordered(ctx, limit, intChan(n), f)
}

var (
	orderedStages   = map[string]stageFunc{"seq": seqStage, "chan": chanStage}
	unorderedStages = map[string]stageFunc{"seq": seqStageUnordered, "chan": chanStageUnordered}
	bothStages      = map[string]stageFunc{"seq": seqStage, "chan": chanStage, "seq-unordered": seqStageUnordered, "chan-unordered": chanStageUnordered}
)

func eachStage(t *testing.T, stages map[string]stageFunc, body func(t *testing.T, run stageFunc)) {
	t.Helper()
	for name, run := range stages {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) { body(t, run) })
		})
	}
}

type stageResult[R any] struct {
	rs  []R
	err error
	pan any
}

func startStage[R any](seq iter.Seq2[R, error], each func(R) bool) <-chan stageResult[R] {
	out := make(chan stageResult[R], 1)
	go func() {
		var d stageResult[R]
		defer func() {
			d.pan = recover()
			out <- d
		}()
		for r, err := range seq {
			if err != nil {
				d.err = err
				return
			}
			d.rs = append(d.rs, r)
			if each != nil && !each(r) {
				return
			}
		}
	}()
	return out
}

func finished[R any](t *testing.T, c <-chan stageResult[R]) stageResult[R] {
	t.Helper()
	select {
	case d := <-c:
		return d
	case <-time.After(stageDeadline):
		t.Fatal("range did not finish")
		return stageResult[R]{}
	}
}

func recv[T any](t *testing.T, ch <-chan T) T {
	t.Helper()
	select {
	case v := <-ch:
		return v
	case <-time.After(stageDeadline):
		t.Fatal("receive timed out")
		var zero T
		return zero
	}
}

func raise(high *atomic.Int32, v int32) {
	for {
		cur := high.Load()
		if v <= cur || high.CompareAndSwap(cur, v) {
			return
		}
	}
}

func panicOf(fn func()) (v any) {
	defer func() { v = recover() }()
	fn()
	return nil
}

func goroutineID() uint64 {
	buf := make([]byte, 64)
	buf = buf[:runtime.Stack(buf, false)]
	id, _, _ := strings.Cut(strings.TrimPrefix(string(buf), "goroutine "), " ")
	n, _ := strconv.ParseUint(id, 10, 64)
	return n
}

type siblings struct {
	running   atomic.Int32
	started   chan struct{}
	cancelled chan struct{}
	hold      chan struct{}
}

func newSiblings() *siblings {
	return &siblings{
		started:   make(chan struct{}, 2),
		cancelled: make(chan struct{}, 2),
		hold:      make(chan struct{}),
	}
}

func (b *siblings) f(ctx context.Context, i int) (int, bool, error) {
	b.running.Add(1)
	defer b.running.Add(-1)
	if i == 0 {
		<-b.started
		<-b.started
		return 0, true, nil
	}
	b.started <- struct{}{}
	<-ctx.Done()
	b.cancelled <- struct{}{}
	<-b.hold
	return 0, false, ctx.Err()
}

func (b *siblings) joinedBeforeReturn(t *testing.T, c <-chan stageResult[int]) stageResult[int] {
	t.Helper()
	recv(t, b.cancelled)
	recv(t, b.cancelled)
	synctest.Wait()
	select {
	case <-c:
		t.Fatal("range returned while children were still running")
	default:
	}
	close(b.hold)
	d := finished(t, c)
	synctest.Wait()
	if n := b.running.Load(); n != 0 {
		t.Fatalf("running = %d after the range returned, want 0", n)
	}
	return d
}

type closeLog struct {
	mu  sync.Mutex
	ids []int
}

func (l *closeLog) add(id int) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ids = append(l.ids, id)
}

func (l *closeLog) snapshot() []int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return slices.Clone(l.ids)
}

type tracked struct {
	id  int
	log *closeLog
}

func (c *tracked) Close() error {
	c.log.add(c.id)
	return nil
}

type panicCloser struct {
	id  int
	log *closeLog
}

func (c *panicCloser) Close() error {
	c.log.add(c.id)
	if c.id == 2 {
		panic("close boom")
	}
	return nil
}

func goroutinesAtMost(base int) int {
	n := runtime.NumGoroutine()
	for i := 0; n > base && i < 100000; i++ {
		runtime.Gosched()
		n = runtime.NumGoroutine()
	}
	return n
}

func TestGroupSeqInputOrderUnderReverseCompletion(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		const n = 4
		gates := make([]chan struct{}, n)
		for i := range gates {
			gates[i] = make(chan struct{})
		}
		got := make(chan int, n)
		seq := run(t.Context(), n, n, func(_ context.Context, i int) (int, bool, error) {
			<-gates[i]
			return i * 10, true, nil
		})
		c := startStage(seq, func(r int) bool {
			got <- r
			return true
		})
		for i := n - 1; i > 0; i-- {
			close(gates[i])
			synctest.Wait()
			if len(got) != 0 {
				t.Fatalf("%d results delivered before the head element finished", len(got))
			}
		}
		close(gates[0])
		d := finished(t, c)
		if want := []int{0, 10, 20, 30}; d.err != nil || !slices.Equal(d.rs, want) {
			t.Fatalf("rs = %v, err = %v, want %v, nil", d.rs, d.err, want)
		}
	})
}

func TestGroupSeqUnorderedCompletionOrder(t *testing.T) {
	eachStage(t, unorderedStages, func(t *testing.T, run stageFunc) {
		const n = 3
		gates := make([]chan struct{}, n)
		for i := range gates {
			gates[i] = make(chan struct{})
		}
		got := make(chan int, n)
		seq := run(t.Context(), n, n, func(_ context.Context, i int) (int, bool, error) {
			<-gates[i]
			return i * 10, true, nil
		})
		c := startStage(seq, func(r int) bool {
			got <- r
			return true
		})
		for _, i := range []int{2, 0, 1} {
			close(gates[i])
			synctest.Wait()
			if len(got) != 1 {
				t.Fatalf("%d results delivered after releasing child %d, want 1", len(got), i)
			}
			if v := <-got; v != i*10 {
				t.Fatalf("delivered %d after releasing child %d, want %d", v, i, i*10)
			}
		}
		if d := finished(t, c); d.err != nil || len(d.rs) != n {
			t.Fatalf("rs = %v, err = %v, want %d results and nil", d.rs, d.err, n)
		}
	})
}

func TestGroupSeqWindowBoundsInFlightChildren(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const limit, total = 3, 40
		var inflight, high atomic.Int32
		gate := make(chan struct{})
		seq := GroupSeq(t.Context(), limit, intSeq(total), func(_ context.Context, i int) (int, bool, error) {
			raise(&high, inflight.Add(1))
			<-gate
			inflight.Add(-1)
			return i, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		if n := inflight.Load(); n != limit {
			t.Fatalf("%d children in flight with a full window, want exactly %d", n, limit)
		}
		for range total {
			gate <- struct{}{}
		}
		d := finished(t, c)
		if len(d.rs) != total || d.err != nil {
			t.Fatalf("len(rs) = %d, err = %v, want %d, nil", len(d.rs), d.err, total)
		}
		if h := high.Load(); h > limit {
			t.Fatalf("children in flight reached %d, limit %d", h, limit)
		}
	})
}

func TestGroupSeqWindowBoundsFinishedUndelivered(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const limit, total = 4, 10000
		var outstanding, high atomic.Int32
		seq := GroupSeq(t.Context(), limit, intSeq(total), func(_ context.Context, i int) (int, bool, error) {
			raise(&high, outstanding.Add(1))
			return i, true, nil
		})
		d := finished(t, startStage(seq, func(int) bool {
			outstanding.Add(-1)
			time.Sleep(time.Millisecond)
			return true
		}))
		if len(d.rs) != total || d.err != nil {
			t.Fatalf("len(rs) = %d, err = %v, want %d, nil", len(d.rs), d.err, total)
		}
		if h := high.Load(); h > limit {
			t.Fatalf("live plus finished-undelivered reached %d, limit %d", h, limit)
		}
	})
}

func TestGroupSeqSlowHeadFillsWindowWithoutDeadlock(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const limit, total = 8, 30
		var entered atomic.Int32
		head := make(chan struct{})
		seq := GroupSeq(t.Context(), limit, intSeq(total), func(_ context.Context, i int) (int, bool, error) {
			entered.Add(1)
			if i == 0 {
				<-head
			}
			return i, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		if n := entered.Load(); n != limit {
			t.Fatalf("%d children started behind an unfinished head, want exactly the window %d", n, limit)
		}
		close(head)
		d := finished(t, c)
		if d.err != nil || !slices.Equal(d.rs, slices.Collect(intSeq(total))) {
			t.Fatalf("rs = %v, err = %v, want 0..%d in order, nil", d.rs, d.err, total-1)
		}
	})
}

func TestGroupSeqUnorderedFirstFailureWinsAndCancelsSiblings(t *testing.T) {
	eachStage(t, unorderedStages, func(t *testing.T, run stageFunc) {
		errFirst := errors.New("first")
		errSibling := errors.New("sibling")
		var sawCancel atomic.Bool
		seq := run(t.Context(), 2, 2, func(ctx context.Context, i int) (int, bool, error) {
			if i == 0 {
				<-ctx.Done()
				sawCancel.Store(true)
				return 0, false, errSibling
			}
			return 0, false, errFirst
		})
		d := finished(t, startStage(seq, nil))
		if !errors.Is(d.err, errFirst) || len(d.rs) != 0 {
			t.Fatalf("rs = %v, err = %v, want none, %v", d.rs, d.err, errFirst)
		}
		if !sawCancel.Load() {
			t.Fatal("sibling did not observe cancellation")
		}
	})
}

func TestGroupSeqChildPanicBecomesPanicError(t *testing.T) {
	eachStage(t, bothStages, func(t *testing.T, run stageFunc) {
		seq := run(t.Context(), 2, 2, func(ctx context.Context, i int) (int, bool, error) {
			if i == 0 {
				panic("kapow")
			}
			<-ctx.Done()
			return 0, false, ctx.Err()
		})
		d := finished(t, startStage(seq, nil))
		pe, ok := errors.AsType[*PanicError](d.err)
		if !ok || pe.Value != "kapow" || len(pe.Stack) == 0 {
			t.Fatalf("err = %#v, want *PanicError holding kapow with a stack", d.err)
		}
	})
}

func TestGroupSeqEarlyStopCancelsAndJoins(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		b := newSiblings()
		c := startStage(run(t.Context(), 3, 3, b.f), func(int) bool { return false })
		d := b.joinedBeforeReturn(t, c)
		if d.err != nil || !slices.Equal(d.rs, []int{0}) || d.pan != nil {
			t.Fatalf("rs = %v, err = %v, panic = %v, want [0], nil, nil", d.rs, d.err, d.pan)
		}
	})
}

func TestGroupSeqLoopBodyPanicCancelsAndJoins(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		b := newSiblings()
		c := startStage(run(t.Context(), 3, 3, b.f), func(int) bool { panic("body") })
		if d := b.joinedBeforeReturn(t, c); d.pan != "body" {
			t.Fatalf("panic = %v, want the loop body's panic value", d.pan)
		}
	})
}

func TestGroupSeqSourcePanicCancelsAndJoins(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		b := newSiblings()
		src := func(yield func(int) bool) {
			for i := range 3 {
				if !yield(i) {
					return
				}
			}
			panic("src")
		}
		c := startStage(GroupSeq(t.Context(), 3, src, b.f), nil)
		if d := b.joinedBeforeReturn(t, c); d.pan != "src" {
			t.Fatalf("panic = %v, want the source's panic value", d.pan)
		}
	})
}

func TestGroupSeqEarlyStopDiscardsLaterFailure(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errLate := errors.New("late")
		started := make(chan struct{}, 2)
		bodyEntered := make(chan struct{})
		sawCancel := make(chan struct{})
		var running atomic.Int32
		seq := run(t.Context(), 3, 3, func(ctx context.Context, i int) (int, bool, error) {
			running.Add(1)
			defer running.Add(-1)
			switch i {
			case 0:
				<-started
				<-started
				return 0, true, nil
			case 1:
				started <- struct{}{}
				<-bodyEntered
				return 0, false, errLate
			default:
				started <- struct{}{}
				<-ctx.Done()
				close(sawCancel)
				return 0, false, ctx.Err()
			}
		})
		d := finished(t, startStage(seq, func(int) bool {
			close(bodyEntered)
			synctest.Wait()
			return false
		}))
		synctest.Wait()
		if d.err != nil || d.pan != nil || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("rs = %v, err = %v, panic = %v, want [0], nil, nil", d.rs, d.err, d.pan)
		}
		if n := running.Load(); n != 0 {
			t.Fatalf("running = %d after the range returned, want 0", n)
		}
	})
}

func TestGroupSeqOrderedFailedElementIsNeverPassedOver(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errHead := errors.New("head")
		finishedTail := make(chan struct{}, 2)
		release := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (int, bool, error) {
			if i == 0 {
				<-release
				return 0, false, errHead
			}
			finishedTail <- struct{}{}
			return i, true, nil
		})
		got := make(chan int, 2)
		c := startStage(seq, func(r int) bool {
			got <- r
			return true
		})
		synctest.Wait()
		if len(finishedTail) != 2 || len(got) != 0 {
			t.Fatalf("tail finished = %d, delivered = %d, want 2, 0 while the head is pending", len(finishedTail), len(got))
		}
		close(release)
		d := finished(t, c)
		if !errors.Is(d.err, errHead) || len(d.rs) != 0 {
			t.Fatalf("rs = %v, err = %v, want none, %v", d.rs, d.err, errHead)
		}
	})
}

func TestGroupSeqClosesUndeliveredInReverseInputOrder(t *testing.T) {
	all := entries[*tracked]()
	for _, name := range []string{"seq", "chan"} {
		t.Run(name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				var log closeLog
				errBoom := errors.New("boom")
				started := make(chan struct{}, 3)
				delivered := make(chan struct{})
				seq := all[name](t.Context(), 5, 5, func(_ context.Context, i int) (*tracked, bool, error) {
					switch i {
					case 0:
						return &tracked{i, &log}, true, nil
					case 1:
						<-delivered
						for range 3 {
							<-started
						}
						return nil, false, errBoom
					default:
						started <- struct{}{}
						return &tracked{i, &log}, true, nil
					}
				})
				d := finished(t, startStage(seq, func(*tracked) bool {
					close(delivered)
					return true
				}))
				if !errors.Is(d.err, errBoom) || len(d.rs) != 1 || d.rs[0].id != 0 {
					t.Fatalf("rs = %v, err = %v, want the result of element 0 and %v", d.rs, d.err, errBoom)
				}
				if want := []int{4, 3, 2}; !slices.Equal(log.snapshot(), want) {
					t.Fatalf("closed = %v, want %v", log.snapshot(), want)
				}
			})
		})
	}
}

func TestGroupSeqClosesUndeliveredOnEarlyStop(t *testing.T) {
	eachEntry(t, func(t *testing.T, run entryFunc[*tracked]) {
		var log closeLog
		started := make(chan struct{}, 2)
		stop := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (*tracked, bool, error) {
			if i == 0 {
				<-started
				<-started
			} else {
				started <- struct{}{}
				<-stop
			}
			return &tracked{i, &log}, true, nil
		})
		d := finished(t, startStage(seq, func(*tracked) bool {
			close(stop)
			return false
		}))
		if d.err != nil || len(d.rs) != 1 || d.rs[0].id != 0 {
			t.Fatalf("rs = %v, err = %v, want the result of element 0 and nil", d.rs, d.err)
		}
		if want := []int{2, 1}; !slices.Equal(log.snapshot(), want) {
			t.Fatalf("closed = %v, want %v", log.snapshot(), want)
		}
	})
}

func TestGroupSeqFilterDropsAreClosedOnce(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const total = 200
		var log closeLog
		seq := GroupSeq(t.Context(), 3, intSeq(total), func(_ context.Context, i int) (*tracked, bool, error) {
			return &tracked{i, &log}, i%2 == 0, nil
		})
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != total/2 {
			t.Fatalf("len(rs) = %d, err = %v, want %d, nil", len(d.rs), d.err, total/2)
		}
		closed := log.snapshot()
		slices.Sort(closed)
		var want []int
		for i := 1; i < total; i += 2 {
			want = append(want, i)
		}
		if !slices.Equal(closed, want) {
			t.Fatalf("closed = %v, want each dropped odd element exactly once", closed)
		}
	})
}

func TestGroupSeqClosePanicPropagatesAfterJoin(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var log closeLog
		errHead := errors.New("head")
		started := make(chan struct{}, 3)
		hold := make(chan struct{})
		seq := GroupSeq(t.Context(), 4, intSeq(4), func(_ context.Context, i int) (*panicCloser, bool, error) {
			if i == 0 {
				<-started
				<-started
				<-started
				return nil, false, errHead
			}
			started <- struct{}{}
			if i == 3 {
				<-hold
			}
			return &panicCloser{id: i, log: &log}, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed %v before the child that ignores cancellation returned", closed)
		}
		close(hold)
		d := finished(t, c)
		if d.pan != "close boom" {
			t.Fatalf("panic = %v, err = %v, want the Close panic to reach the caller", d.pan, d.err)
		}
		if want := []int{3, 2, 1}; !slices.Equal(log.snapshot(), want) {
			t.Fatalf("closed = %v, want %v: a panicking Close must not skip the others", log.snapshot(), want)
		}
	})
}

func TestGroupSeqEmptySourceYieldsNothing(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		var calls atomic.Int32
		seq := run(t.Context(), 2, 0, func(_ context.Context, i int) (int, bool, error) {
			calls.Add(1)
			return i, true, nil
		})
		d := finished(t, startStage(seq, nil))
		if len(d.rs) != 0 || d.err != nil || calls.Load() != 0 {
			t.Fatalf("rs = %v, err = %v, calls = %d, want none, nil, 0", d.rs, d.err, calls.Load())
		}
	})
}

func TestGroupSeqLimitMustBePositive(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const want = "logos: group stage limit must be positive"
		for _, limit := range []int{0, -1} {
			var pulled atomic.Int32
			src := func(yield func(int) bool) {
				pulled.Add(1)
				yield(0)
			}
			ch := make(chan int, 1)
			ch <- 0
			f := func(_ context.Context, i int) (int, bool, error) { return i, true, nil }
			for name, seq := range map[string]iter.Seq2[int, error]{
				"seq":            GroupSeq(t.Context(), limit, src, f),
				"seq unordered":  GroupSeqUnordered(t.Context(), limit, src, f),
				"chan":           GroupChan(t.Context(), limit, ch, f),
				"chan unordered": GroupChanUnordered(t.Context(), limit, ch, f),
			} {
				if d := finished(t, startStage(seq, nil)); d.pan != want {
					t.Errorf("%s with limit %d: panic = %v, want %q", name, limit, d.pan, want)
				}
			}
			if pulled.Load() != 0 || len(ch) != 1 {
				t.Errorf("limit %d touched a source: pulled = %d, queued = %d, want 0, 1", limit, pulled.Load(), len(ch))
			}
		}
	})
}

func TestGroupSeqParentCancelledBeforeRange(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		var calls atomic.Int32
		seq := run(ctx, 2, 4, func(_ context.Context, i int) (int, bool, error) {
			calls.Add(1)
			return i, true, nil
		})
		d := finished(t, startStage(seq, nil))
		if !errors.Is(d.err, context.Canceled) || len(d.rs) != 0 || calls.Load() != 0 {
			t.Fatalf("rs = %v, err = %v, calls = %d, want none, context.Canceled, 0", d.rs, d.err, calls.Load())
		}
	})
}

func TestGroupSeqParentCancelledDuringRange(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		seq := run(ctx, 2, 100, func(_ context.Context, i int) (int, bool, error) { return i, true, nil })
		d := finished(t, startStage(seq, func(int) bool {
			cancel()
			return true
		}))
		if !errors.Is(d.err, context.Canceled) || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("rs = %v, err = %v, want [0], context.Canceled", d.rs, d.err)
		}
	})
}

func TestGroupSeqLeavesNoGoroutineBehind(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		base := runtime.NumGoroutine()
		errBoom := errors.New("boom")
		blockedAfterHead := func(ctx context.Context, i int) (int, bool, error) {
			if i == 0 {
				return i, true, nil
			}
			<-ctx.Done()
			return 0, false, ctx.Err()
		}
		exits := map[string]func() <-chan stageResult[int]{
			"exhaustion": func() <-chan stageResult[int] {
				return startStage(seqStage(t.Context(), 3, 20, func(_ context.Context, i int) (int, bool, error) { return i, true, nil }), nil)
			},
			"failure": func() <-chan stageResult[int] {
				return startStage(seqStage(t.Context(), 3, 20, func(ctx context.Context, i int) (int, bool, error) {
					if i == 1 {
						return 0, false, errBoom
					}
					return blockedAfterHead(ctx, i)
				}), nil)
			},
			"break": func() <-chan stageResult[int] {
				return startStage(seqStage(t.Context(), 3, 20, blockedAfterHead), func(int) bool { return false })
			},
		}
		for name, start := range exits {
			finished(t, start())
			synctest.Wait()
			if got := goroutinesAtMost(base); got > base {
				t.Fatalf("after %s: %d goroutines, want at most %d", name, got, base)
			}
		}
	})
}

func TestGroupSeqSourceAndBodyRunOnRangingGoroutine(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var srcIDs, bodyIDs, childIDs []uint64
		var mu sync.Mutex
		var ranging uint64
		src := func(yield func(int) bool) {
			for i := range 6 {
				srcIDs = append(srcIDs, goroutineID())
				if !yield(i) {
					return
				}
			}
		}
		seq := GroupSeq(t.Context(), 2, src, func(_ context.Context, i int) (int, bool, error) {
			mu.Lock()
			defer mu.Unlock()
			childIDs = append(childIDs, goroutineID())
			return i, true, nil
		})
		done := make(chan struct{})
		go func() {
			defer close(done)
			ranging = goroutineID()
			for range seq {
				bodyIDs = append(bodyIDs, goroutineID())
			}
		}()
		recv(t, done)
		for _, ids := range [][]uint64{srcIDs, bodyIDs} {
			if len(ids) != 6 || slices.ContainsFunc(ids, func(id uint64) bool { return id != ranging }) {
				t.Fatalf("source or body ran on %v, want six runs on goroutine %d", ids, ranging)
			}
		}
		if len(childIDs) != 6 || slices.Contains(childIDs, ranging) {
			t.Fatalf("children ran on %v, want six runs away from goroutine %d", childIDs, ranging)
		}
	})
}

func TestGroupChanLoopBodyRunsOnRangingGoroutine(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var bodyIDs []uint64
		var ranging uint64
		seq := GroupChan(t.Context(), 2, intChan(6), func(_ context.Context, i int) (int, bool, error) { return i, true, nil })
		done := make(chan struct{})
		go func() {
			defer close(done)
			ranging = goroutineID()
			for range seq {
				bodyIDs = append(bodyIDs, goroutineID())
			}
		}()
		recv(t, done)
		if len(bodyIDs) != 6 || slices.ContainsFunc(bodyIDs, func(id uint64) bool { return id != ranging }) {
			t.Fatalf("body ran on %v, want six runs on goroutine %d", bodyIDs, ranging)
		}
	})
}

func TestGroupChanReceivesOnRangingGoroutine(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ch := make(chan int)
		seq := GroupChan(t.Context(), 2, ch, func(_ context.Context, i int) (int, bool, error) { return i, true, nil })
		var ranging uint64
		done := make(chan struct{})
		go func() {
			defer close(done)
			ranging = goroutineID()
			for range seq {
			}
		}()
		synctest.Wait()
		stack := goroutineStack(ranging)
		close(ch)
		recv(t, done)
		if !strings.Contains(stack, ").fromChan(") {
			t.Fatalf("goroutine %d parked outside fromChan:\n%s", ranging, stack)
		}
	})
}

func goroutineStack(id uint64) string {
	buf := make([]byte, 1<<20)
	buf = buf[:runtime.Stack(buf, true)]
	head := "goroutine " + strconv.FormatUint(id, 10) + " "
	for g := range strings.SplitSeq(string(buf), "\n\n") {
		if strings.HasPrefix(g, head) {
			return g
		}
	}
	return ""
}

func TestGroupChanDeliversWhileChannelIsQuiet(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		ch := make(chan int)
		seq := GroupChan(t.Context(), 2, ch, func(_ context.Context, v int) (int, bool, error) { return v * 2, true, nil })
		got := make(chan int, 1)
		c := startStage(seq, func(r int) bool {
			got <- r
			return true
		})
		ch <- 21
		synctest.Wait()
		if len(got) != 1 {
			t.Fatal("the finished result was not delivered while the channel stayed quiet")
		}
		if v := <-got; v != 42 {
			t.Fatalf("delivered %d, want 42", v)
		}
		close(ch)
		if d := finished(t, c); d.err != nil || !slices.Equal(d.rs, []int{42}) {
			t.Fatalf("rs = %v, err = %v, want [42], nil", d.rs, d.err)
		}
	})
}

func TestGroupChanEndsWithChildErrorWhileChannelIsQuiet(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		errBoom := errors.New("boom")
		ch := make(chan int)
		fail := make(chan struct{})
		seq := GroupChan(t.Context(), 2, ch, func(_ context.Context, v int) (int, bool, error) {
			<-fail
			return 0, false, errBoom
		})
		c := startStage(seq, nil)
		ch <- 1
		close(fail)
		synctest.Wait()
		select {
		case d := <-c:
			if !errors.Is(d.err, errBoom) || len(d.rs) != 0 {
				t.Fatalf("rs = %v, err = %v, want none, %v", d.rs, d.err, errBoom)
			}
		default:
			t.Fatal("the statement did not end after a child failed while the channel stayed open and quiet")
		}
	})
}

func TestGroupSeqRunsFOnAtMostLimitWorkers(t *testing.T) {
	for _, c := range []struct{ limit, total int }{{4, 2000}, {16384, 5}} {
		eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
			var mu sync.Mutex
			var peak atomic.Int32
			workers := map[uint64]bool{}
			base := runtime.NumGoroutine()
			seq := run(t.Context(), c.limit, c.total, func(_ context.Context, i int) (int, bool, error) {
				raise(&peak, int32(runtime.NumGoroutine()))
				mu.Lock()
				defer mu.Unlock()
				workers[goroutineID()] = true
				return i, true, nil
			})
			d := finished(t, startStage(seq, nil))
			if d.err != nil || len(d.rs) != c.total {
				t.Fatalf("len(rs) = %d, err = %v, want %d, nil", len(d.rs), d.err, c.total)
			}
			most := min(c.limit, c.total)
			if len(workers) < 1 || len(workers) > most {
				t.Fatalf("f ran on %d goroutines over %d elements with limit %d, want 1 to %d", len(workers), c.total, c.limit, most)
			}
			if grew := int(peak.Load()) - base; grew > most+2 {
				t.Fatalf("%d goroutines alive beyond the %d at the start with limit %d and %d elements, want at most %d", grew, base, c.limit, c.total, most+2)
			}
		})
	}
}

func TestGroupSeqLeavesNoWorkerAliveAfterAnyExit(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		const limit, total = 4, 200
		errBoom := errors.New("boom")
		okWork := func(i int) (int, error) { return i, nil }
		stopAfter := func(n int) func(int) bool { return func(r int) bool { return r < n } }
		exits := map[string]struct {
			src  iter.Seq[int]
			work func(i int) (int, error)
			each func(int) bool
		}{
			"exhaustion": {intSeq(total), okWork, nil},
			"break":      {intSeq(total), okWork, stopAfter(20)},
			"error": {intSeq(total), func(i int) (int, error) {
				if i == 50 {
					return 0, errBoom
				}
				return i, nil
			}, nil},
			"child panic": {intSeq(total), func(i int) (int, error) {
				if i == 50 {
					panic("kapow")
				}
				return i, nil
			}, nil},
			"body panic": {intSeq(total), okWork, func(r int) bool {
				if r == 20 {
					panic("body")
				}
				return true
			}},
			"source panic": {func(yield func(int) bool) {
				for i := range 60 {
					if !yield(i) {
						return
					}
				}
				panic("src")
			}, okWork, nil},
		}
		base := runtime.NumGoroutine()
		for name, sc := range exits {
			var entered atomic.Int32
			allIn := make(chan struct{})
			seq := GroupSeq(t.Context(), limit, sc.src, func(_ context.Context, i int) (int, bool, error) {
				if i < limit {
					if entered.Add(1) == limit {
						close(allIn)
					}
					<-allIn
				}
				v, err := sc.work(i)
				return v, true, err
			})
			finished(t, startStage(seq, sc.each))
			synctest.Wait()
			if got := goroutinesAtMost(base); got > base {
				t.Fatalf("after %s: %d goroutines, want at most %d", name, got, base)
			}
		}
	})
}

func TestGroupSeqAllocationsDoNotGrowWithElements(t *testing.T) {
	const small, large = 2000, 20000
	keeps := map[string]func(int) bool{
		"kept":    func(int) bool { return true },
		"dropped": func(i int) bool { return i%2 == 0 },
	}
	for name, run := range orderedStages {
		for shape, keep := range keeps {
			t.Run(name+"/"+shape, func(t *testing.T) {
				allocs := func(n int) float64 {
					return testing.AllocsPerRun(5, func() {
						for range run(t.Context(), 8, n, func(_ context.Context, i int) (int, bool, error) { return i + 1000, keep(i), nil }) {
						}
					})
				}
				low, high := allocs(small), allocs(large)
				if per := (high - low) / (large - small); per > 0.05 {
					t.Fatalf("%.2f allocations per additional element (%.0f at %d, %.0f at %d), want at most 0.05", per, low, small, high, large)
				}
			})
		}
	}
}

func TestGroupSeqOrderedErrorIsLowestIndexAfterDeliveredPrefix(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errA := errors.New("A: element 1")
		errB := errors.New("B: element 2")
		gates := [2]chan struct{}{make(chan struct{}), make(chan struct{})}
		seq := run(t.Context(), 4, 4, func(_ context.Context, i int) (int, bool, error) {
			switch i {
			case 0:
				<-gates[0]
				return 100, true, nil
			case 1:
				<-gates[1]
				return 0, false, errA
			case 2:
				return 0, false, errB
			}
			return i, true, nil
		})
		got := make(chan int, 4)
		c := startStage(seq, func(r int) bool {
			got <- r
			return true
		})
		synctest.Wait()
		if len(got) != 0 {
			t.Fatalf("%d results delivered while the head element was pending", len(got))
		}
		close(gates[1])
		synctest.Wait()
		close(gates[0])
		d := finished(t, c)
		if !errors.Is(d.err, errA) || !slices.Equal(d.rs, []int{100}) {
			t.Fatalf("rs = %v, err = %v, want [100], %v", d.rs, d.err, errA)
		}
	})
}

func TestGroupSeqOrderedFailureCancelsOnlyWhenItReachesTheHead(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errBoom := errors.New("boom")
		gate := make(chan struct{})
		headSaw := make(chan error, 1)
		tailSawCancel := make(chan struct{})
		tailEntered := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(ctx context.Context, i int) (int, bool, error) {
			switch i {
			case 0:
				<-gate
				headSaw <- ctx.Err()
				return 0, true, nil
			case 1:
				<-tailEntered
				return 0, false, errBoom
			}
			close(tailEntered)
			<-ctx.Done()
			close(tailSawCancel)
			return 0, false, ctx.Err()
		})
		c := startStage(seq, nil)
		synctest.Wait()
		select {
		case <-tailSawCancel:
			t.Fatal("the siblings were cancelled before the failure reached the head")
		default:
		}
		close(gate)
		d := finished(t, c)
		if err := recv(t, headSaw); err != nil {
			t.Fatalf("the head element saw %v, want no cancellation", err)
		}
		recv(t, tailSawCancel)
		if !errors.Is(d.err, errBoom) || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("rs = %v, err = %v, want [0], %v", d.rs, d.err, errBoom)
		}
	})
}

func TestGroupSeqOrderedFailureAtHeadCancelsLaterSiblings(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errFirst := errors.New("first")
		var sawCancel atomic.Int32
		entered := make(chan struct{}, 2)
		seq := run(t.Context(), 3, 3, func(ctx context.Context, i int) (int, bool, error) {
			if i == 0 {
				<-entered
				<-entered
				return 0, false, errFirst
			}
			entered <- struct{}{}
			<-ctx.Done()
			sawCancel.Add(1)
			return 0, false, ctx.Err()
		})
		d := finished(t, startStage(seq, nil))
		synctest.Wait()
		if !errors.Is(d.err, errFirst) || len(d.rs) != 0 || sawCancel.Load() != 2 {
			t.Fatalf("rs = %v, err = %v, cancelled = %d, want none, %v, 2", d.rs, d.err, sawCancel.Load(), errFirst)
		}
	})
}

func TestGroupSeqOrderedEarlierPanicBeatsLaterError(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errLater := errors.New("later")
		gate := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (int, bool, error) {
			switch i {
			case 1:
				<-gate
				panic("kapow")
			case 2:
				return 0, false, errLater
			}
			return i, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		close(gate)
		d := finished(t, c)
		pe, ok := errors.AsType[*PanicError](d.err)
		if !ok || pe.Value != "kapow" || len(pe.Stack) == 0 || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("rs = %v, err = %v, want [0] and *PanicError holding kapow with a stack", d.rs, d.err)
		}
	})
}

func TestGroupSeqOrderedBreakBeforeFailureDiscardsIt(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		errLater := errors.New("later")
		gate := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (int, bool, error) {
			switch i {
			case 0:
				<-gate
				return 7, true, nil
			case 1:
				return 0, false, errLater
			}
			return i, true, nil
		})
		c := startStage(seq, func(int) bool { return false })
		synctest.Wait()
		close(gate)
		d := finished(t, c)
		if d.err != nil || d.pan != nil || !slices.Equal(d.rs, []int{7}) {
			t.Fatalf("rs = %v, err = %v, panic = %v, want [7], nil, nil", d.rs, d.err, d.pan)
		}
	})
}

func TestGroupSeqOrderedPanicWaitsForTheHeadLikeAnError(t *testing.T) {
	eachStage(t, orderedStages, func(t *testing.T, run stageFunc) {
		gate := make(chan struct{})
		seq := run(t.Context(), 2, 2, func(_ context.Context, i int) (int, bool, error) {
			if i == 1 {
				panic("kapow")
			}
			<-gate
			return 7, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		close(gate)
		d := finished(t, c)
		pe, ok := errors.AsType[*PanicError](d.err)
		if !ok || pe.Value != "kapow" || len(pe.Stack) == 0 || !slices.Equal(d.rs, []int{7}) {
			t.Fatalf("rs = %v, err = %v, want [7] and *PanicError holding kapow with a stack", d.rs, d.err)
		}
	})
}

func TestGroupSeqOrderedStopsPullingAfterFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		errBoom := errors.New("boom")
		var calls atomic.Int32
		proceed := make(chan struct{})
		gate := make(chan struct{})
		src := func(yield func(int) bool) {
			for i := range 3 {
				if i == 2 {
					<-proceed
				}
				if !yield(i) {
					return
				}
			}
		}
		seq := GroupSeq(t.Context(), 4, src, func(_ context.Context, i int) (int, bool, error) {
			calls.Add(1)
			switch i {
			case 0:
				<-gate
				return 0, true, nil
			case 1:
				return 0, false, errBoom
			}
			return i, true, nil
		})
		c := startStage(seq, nil)
		synctest.Wait()
		close(proceed)
		synctest.Wait()
		close(gate)
		d := finished(t, c)
		if n := calls.Load(); n != 2 || !errors.Is(d.err, errBoom) || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("f ran %d times, rs = %v, err = %v, want 2, [0], %v", n, d.rs, d.err, errBoom)
		}
	})
}

func TestGroupChanOrderedStopsReceivingAfterFailure(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		errBoom := errors.New("boom")
		gate := make(chan struct{})
		ch := make(chan int)
		seq := GroupChan(t.Context(), 4, ch, func(_ context.Context, i int) (int, bool, error) {
			switch i {
			case 0:
				<-gate
				return 0, true, nil
			case 1:
				return 0, false, errBoom
			}
			return i, true, nil
		})
		c := startStage(seq, nil)
		ch <- 0
		ch <- 1
		synctest.Wait()
		select {
		case ch <- 2:
			t.Fatal("a source element was taken after a failure was known")
		default:
		}
		close(gate)
		d := finished(t, c)
		if !errors.Is(d.err, errBoom) || !slices.Equal(d.rs, []int{0}) {
			t.Fatalf("rs = %v, err = %v, want [0], %v", d.rs, d.err, errBoom)
		}
	})
}
