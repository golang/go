// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"context"
	"errors"
	"slices"
	"testing"
	"testing/synctest"
)

func TestGroupStageClosePanicPropagatesAfterJoinOnEveryEntry(t *testing.T) {
	eachEntry(t, func(t *testing.T, run entryFunc[*panicCloser]) {
		var log closeLog
		errHead := errors.New("head")
		var release [3]chan struct{}
		for i := range release {
			release[i] = make(chan struct{})
		}
		hold := make(chan struct{})
		gate := make(chan struct{})
		seq := run(t.Context(), 4, 4, func(_ context.Context, i int) (*panicCloser, bool, error) {
			switch i {
			case 1:
				<-release[i]
				return nil, false, errHead
			case 3:
				<-hold
			default:
				<-release[i]
			}
			return &panicCloser{id: i, log: &log}, true, nil
		})
		c := startStage(seq, func(*panicCloser) bool {
			<-gate
			return true
		})
		synctest.Wait()
		for _, i := range []int{0, 1, 2} {
			close(release[i])
			synctest.Wait()
		}
		close(gate)
		synctest.Wait()
		if closed := log.snapshot(); len(closed) != 0 {
			t.Fatalf("closed %v before the child that ignores cancellation returned", closed)
		}
		close(hold)
		d := finished(t, c)
		if d.pan != "close boom" {
			t.Fatalf("panic = %v, err = %v, want the Close panic to reach the caller", d.pan, d.err)
		}
		if want := []int{3, 2}; !slices.Equal(log.snapshot(), want) {
			t.Fatalf("closed = %v, want %v: a panicking Close must not skip the others", log.snapshot(), want)
		}
	})
}
