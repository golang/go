// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"slices"
	"testing"
	"testing/synctest"
)

type plain struct {
	id  int
	log *closeLog
}

type resultKind[R any, D Discarder[R]] struct {
	value  func(log *closeLog, id int) R
	empty  func() R
	closes func(id int) bool
}

func (k resultKind[R, D]) closers(ids ...int) []int {
	return slices.DeleteFunc(slices.Clone(ids), func(id int) bool { return !k.closes(id) })
}

var (
	closerKind = resultKind[*tracked, CloseResult[*tracked]]{
		value:  func(log *closeLog, id int) *tracked { return &tracked{id, log} },
		empty:  func() *tracked { return nil },
		closes: func(int) bool { return true },
	}
	plainKind = resultKind[*plain, LeaveResultOpen[*plain]]{
		value:  func(log *closeLog, id int) *plain { return &plain{id, log} },
		empty:  func() *plain { return nil },
		closes: func(int) bool { return false },
	}
	interfaceKind = resultKind[any, CloseResultIfCloser[any]]{
		value: func(log *closeLog, id int) any {
			if id%3 != 0 {
				return &tracked{id, log}
			}
			return &plain{id, log}
		},
		empty:  func() any { return (*tracked)(nil) },
		closes: func(id int) bool { return id%3 != 0 },
	}
)

func TestGroupStageNeverClosesADroppedValueItDoesNotOwn(t *testing.T) {
	t.Run("closer", func(t *testing.T) { neverClosesDroppedNotOwned(t, closerKind) })
	t.Run("non-closer", func(t *testing.T) { neverClosesDroppedNotOwned(t, plainKind) })
	t.Run("interface", func(t *testing.T) { neverClosesDroppedNotOwned(t, interfaceKind) })
}

func neverClosesDroppedNotOwned[R any, D Discarder[R]](t *testing.T, k resultKind[R, D]) {
	eachEntry[R, D](t, func(t *testing.T, run entryFunc[R]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (R, bool, error) {
			if i%2 == 0 {
				return k.value(&log, i), true, nil
			}
			return k.value(&log, i), false, ErrNotOwned
		})
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != 10 || d.pan != nil {
			t.Fatalf("len(rs) = %d, err = %v, panic = %v, want 10, nil, nil: the not-owned mark must not reach the caller", len(d.rs), d.err, d.pan)
		}
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed = %v, want none: the caller keeps every dropped value", closed)
		}
	})
}

func TestGroupStageNeverClosesAnEmptyNotOwnedDrop(t *testing.T) {
	t.Run("closer", func(t *testing.T) { neverClosesEmptyNotOwned(t, closerKind) })
	t.Run("non-closer", func(t *testing.T) { neverClosesEmptyNotOwned(t, plainKind) })
	t.Run("interface", func(t *testing.T) { neverClosesEmptyNotOwned(t, interfaceKind) })
}

func neverClosesEmptyNotOwned[R any, D Discarder[R]](t *testing.T, k resultKind[R, D]) {
	eachEntry[R, D](t, func(t *testing.T, run entryFunc[R]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (R, bool, error) {
			if i%2 == 0 {
				return k.value(&log, i), true, nil
			}
			return k.empty(), false, ErrNotOwned
		})
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != 10 || d.pan != nil {
			t.Fatalf("len(rs) = %d, err = %v, panic = %v, want 10, nil, nil", len(d.rs), d.err, d.pan)
		}
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed = %v, want none", closed)
		}
	})
}

func TestGroupStageClosesAnOwnedDroppedValueOnce(t *testing.T) {
	t.Run("closer", func(t *testing.T) { closesOwnedDroppedOnce(t, closerKind) })
	t.Run("non-closer", func(t *testing.T) { closesOwnedDroppedOnce(t, plainKind) })
	t.Run("interface", func(t *testing.T) { closesOwnedDroppedOnce(t, interfaceKind) })
}

func closesOwnedDroppedOnce[R any, D Discarder[R]](t *testing.T, k resultKind[R, D]) {
	eachEntry[R, D](t, func(t *testing.T, run entryFunc[R]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (R, bool, error) {
			return k.value(&log, i), i%2 == 0, nil
		})
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != 10 {
			t.Fatalf("len(rs) = %d, err = %v, want 10, nil", len(d.rs), d.err)
		}
		closed := log.snapshot()
		slices.Sort(closed)
		want := k.closers(1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
		if !slices.Equal(closed, want) {
			t.Fatalf("closed = %v, want %v: each dropped odd element that implements io.Closer, once", closed, want)
		}
	})
}

func TestGroupStageNeverClosesAnUndeliveredResultItDoesNotOwn(t *testing.T) {
	t.Run("closer", func(t *testing.T) { neverClosesUndeliveredNotOwned(t, closerKind) })
	t.Run("non-closer", func(t *testing.T) { neverClosesUndeliveredNotOwned(t, plainKind) })
	t.Run("interface", func(t *testing.T) { neverClosesUndeliveredNotOwned(t, interfaceKind) })
}

func neverClosesUndeliveredNotOwned[R any, D Discarder[R]](t *testing.T, k resultKind[R, D]) {
	eachEntry[R, D](t, func(t *testing.T, run entryFunc[R]) {
		var log closeLog
		hold := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (R, bool, error) {
			<-hold
			return k.value(&log, i), true, ErrNotOwned
		})
		c := startStage(seq, func(R) bool { return false })
		synctest.Wait()
		close(hold)
		d := finished(t, c)
		if d.err != nil || len(d.rs) != 1 || d.pan != nil {
			t.Fatalf("len(rs) = %d, err = %v, panic = %v, want 1, nil, nil", len(d.rs), d.err, d.pan)
		}
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed = %v, want none: the finished results are not owned", closed)
		}
	})
}

func TestGroupStageClosesAnOwnedUndeliveredResult(t *testing.T) {
	t.Run("closer", func(t *testing.T) { closesOwnedUndelivered(t, closerKind) })
	t.Run("non-closer", func(t *testing.T) { closesOwnedUndelivered(t, plainKind) })
	t.Run("interface", func(t *testing.T) { closesOwnedUndelivered(t, interfaceKind) })
}

func closesOwnedUndelivered[R any, D Discarder[R]](t *testing.T, k resultKind[R, D]) {
	synctest.Test(t, func(t *testing.T) {
		const n = 6
		var log closeLog
		started := make(chan struct{}, n-1)
		seq := GroupSeq[int, R, D](t.Context(), n, intSeq(n), func(_ context.Context, i int) (R, bool, error) {
			switch i {
			case 0:
				for range n - 1 {
					<-started
				}
				return k.value(&log, i), true, nil
			case 1:
				started <- struct{}{}
				return k.value(&log, i), true, ErrNotOwned
			default:
				started <- struct{}{}
				return k.value(&log, i), true, nil
			}
		})
		d := finished(t, startStage(seq, func(R) bool { return false }))
		if d.err != nil || len(d.rs) != 1 {
			t.Fatalf("len(rs) = %d, err = %v, want 1, nil", len(d.rs), d.err)
		}
		if want := k.closers(5, 4, 3, 2); !slices.Equal(log.snapshot(), want) {
			t.Fatalf("closed = %v, want %v: only the owned undelivered results that implement io.Closer, in reverse input order", log.snapshot(), want)
		}
	})
}

func TestGroupSeqNeverClosesASourceElement(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		var log closeLog
		src := func(yield func(*tracked) bool) {
			for i := range 6 {
				if !yield(&tracked{i, &log}) {
					return
				}
			}
		}
		seq := GroupSeq[*tracked, int, LeaveResultOpen[int]](t.Context(), 1, src, func(_ context.Context, e *tracked) (int, bool, error) {
			return e.id, true, nil
		})
		d := finished(t, startStage(seq, func(int) bool { return false }))
		if d.err != nil || len(d.rs) != 1 {
			t.Fatalf("len(rs) = %d, err = %v, want 1, nil", len(d.rs), d.err)
		}
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed = %v, want none: a source element is never owned by the stage", closed)
		}
	})
}
