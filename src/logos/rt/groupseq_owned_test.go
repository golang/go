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

func TestGroupStageNeverClosesADroppedValueItDoesNotOwn(t *testing.T) {
	eachEntry(t, func(t *testing.T, run entryFunc[*tracked]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (*tracked, bool, error) {
			if i%2 == 0 {
				return &tracked{i, &log}, true, nil
			}
			return &tracked{i, &log}, false, ErrNotOwned
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
	eachEntry(t, func(t *testing.T, run entryFunc[*tracked]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (*tracked, bool, error) {
			if i%2 == 0 {
				return &tracked{i, &log}, true, nil
			}
			return nil, false, ErrNotOwned
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
	eachEntry(t, func(t *testing.T, run entryFunc[*tracked]) {
		var log closeLog
		seq := run(t.Context(), 3, 20, func(_ context.Context, i int) (*tracked, bool, error) {
			return &tracked{i, &log}, i%2 == 0, nil
		})
		d := finished(t, startStage(seq, nil))
		if d.err != nil || len(d.rs) != 10 {
			t.Fatalf("len(rs) = %d, err = %v, want 10, nil", len(d.rs), d.err)
		}
		closed := log.snapshot()
		slices.Sort(closed)
		want := []int{1, 3, 5, 7, 9, 11, 13, 15, 17, 19}
		if !slices.Equal(closed, want) {
			t.Fatalf("closed = %v, want each dropped odd element once", closed)
		}
	})
}

func TestGroupStageNeverClosesAnUndeliveredResultItDoesNotOwn(t *testing.T) {
	eachEntry(t, func(t *testing.T, run entryFunc[*tracked]) {
		var log closeLog
		hold := make(chan struct{})
		seq := run(t.Context(), 3, 3, func(_ context.Context, i int) (*tracked, bool, error) {
			<-hold
			return &tracked{i, &log}, true, ErrNotOwned
		})
		c := startStage(seq, func(*tracked) bool { return false })
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
	synctest.Test(t, func(t *testing.T) {
		var log closeLog
		started := make(chan struct{}, 2)
		seq := GroupSeq(t.Context(), 3, intSeq(3), func(_ context.Context, i int) (*tracked, bool, error) {
			switch i {
			case 0:
				<-started
				<-started
				return &tracked{i, &log}, true, nil
			case 1:
				started <- struct{}{}
				return &tracked{i, &log}, true, ErrNotOwned
			default:
				started <- struct{}{}
				return &tracked{i, &log}, true, nil
			}
		})
		d := finished(t, startStage(seq, func(*tracked) bool { return false }))
		if d.err != nil || len(d.rs) != 1 {
			t.Fatalf("len(rs) = %d, err = %v, want 1, nil", len(d.rs), d.err)
		}
		if want := []int{2}; !slices.Equal(log.snapshot(), want) {
			t.Fatalf("closed = %v, want %v: only the owned undelivered result", log.snapshot(), want)
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
		seq := GroupSeq(t.Context(), 1, src, func(_ context.Context, e *tracked) (int, bool, error) {
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
