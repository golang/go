// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rt

import (
	"cmp"
	"context"
	"errors"
	"io"
	"iter"
	"slices"
	"sync"
	"sync/atomic"
)

var ErrNotOwned = errors.New("logos: group stage result is not owned by the stage")

// Discarder disposes of a finished result that the stage owns and does not
// deliver. The lowering names one of the three implementations below as the
// last type argument of a group stage entry.
type Discarder[R any] interface {
	Discard(R)
}

// CloseResult closes every discarded result. The lowering picks it when the
// result type implements io.Closer.
type CloseResult[R io.Closer] struct{}

func (CloseResult[R]) Discard(r R) {
	_ = r.Close()
}

// CloseResultIfCloser closes a discarded result whose dynamic type
// implements io.Closer. The lowering picks it when the result type is an
// interface or a type parameter.
type CloseResultIfCloser[R any] struct{}

func (CloseResultIfCloser[R]) Discard(r R) {
	if c, ok := any(r).(io.Closer); ok {
		_ = c.Close()
	}
}

// LeaveResultOpen leaves every discarded result open. The lowering picks it
// when the result type cannot implement io.Closer.
type LeaveResultOpen[R any] struct{}

func (LeaveResultOpen[R]) Discard(R) {}

func GroupSeq[S, R any, D Discarder[R]](ctx context.Context, limit int, src iter.Seq[S], f func(context.Context, S) (R, bool, error)) iter.Seq2[R, error] {
	s := newGroupStage[S, R, D](limit, f)
	return s.ranged(ctx, func() { s.fromSeq(src) })
}

func GroupSeqUnordered[S, R any, D Discarder[R]](ctx context.Context, limit int, src iter.Seq[S], f func(context.Context, S) (R, bool, error)) iter.Seq2[R, error] {
	s := newGroupStage[S, R, D](limit, f)
	s.completion = true
	return s.ranged(ctx, func() { s.fromSeq(src) })
}

func GroupChan[S, R any, D Discarder[R]](ctx context.Context, limit int, src <-chan S, f func(context.Context, S) (R, bool, error)) iter.Seq2[R, error] {
	s := newGroupStage[S, R, D](limit, f)
	return s.ranged(ctx, func() { s.fromChan(src) })
}

func GroupChanUnordered[S, R any, D Discarder[R]](ctx context.Context, limit int, src <-chan S, f func(context.Context, S) (R, bool, error)) iter.Seq2[R, error] {
	s := newGroupStage[S, R, D](limit, f)
	s.completion = true
	return s.ranged(ctx, func() { s.fromChan(src) })
}

type groupItem[R any] struct {
	idx     int
	keep    bool
	unowned bool
	r       R
	err     error
}

type groupJob[S any] struct {
	idx int
	v   S
}

type groupStage[S, R any, D Discarder[R]] struct {
	limit      int
	f          func(context.Context, S) (R, bool, error)
	completion bool
	used       atomic.Bool
	failKnown  atomic.Bool

	yield   func(R, error) bool
	gdone   <-chan struct{}
	g       *Group
	cancel  context.CancelFunc
	wake    chan struct{}
	spawned int
	started int
	held    int
	stopped bool
	joined  bool
	err     error
	headErr error

	mu     sync.Mutex
	cond   sync.Cond
	jobs   []groupJob[S]
	jhead  int
	idle   int
	closed bool
	buf    map[int]groupItem[R]
	done   int
	next   int
}

func newGroupStage[S, R any, D Discarder[R]](limit int, f func(context.Context, S) (R, bool, error)) *groupStage[S, R, D] {
	return &groupStage[S, R, D]{limit: limit, f: f}
}

func (s *groupStage[S, R, D]) ranged(ctx context.Context, drive func()) iter.Seq2[R, error] {
	return func(yield func(R, error) bool) {
		s.start(ctx, yield)
		defer s.join()
		if s.live() {
			drive()
			s.drain()
		}
		s.conclude(ctx)
	}
}

func (s *groupStage[S, R, D]) start(ctx context.Context, yield func(R, error) bool) {
	if s.limit <= 0 {
		panic("logos: group stage limit must be positive")
	}
	if s.used.Swap(true) {
		panic("logos: group stage sequence ranged twice")
	}
	s.yield = yield
	segCtx, cancel := context.WithCancel(ctx)
	g, gctx := NewGroup(segCtx, 0)
	s.g, s.cancel, s.gdone = g, cancel, gctx.Done()
	s.wake = make(chan struct{}, 1)
	s.cond.L = &s.mu
	s.buf = make(map[int]groupItem[R])
}

func (s *groupStage[S, R, D]) live() bool {
	if s.stopped || s.headErr != nil {
		return false
	}
	select {
	case <-s.gdone:
		return false
	default:
		return true
	}
}

func (s *groupStage[S, R, D]) pulling() bool {
	return s.live() && !s.failKnown.Load()
}

func (s *groupStage[S, R, D]) fromSeq(src iter.Seq[S]) {
	src(func(v S) bool {
		if !s.slot() {
			return false
		}
		s.spawn(v)
		return true
	})
}

func (s *groupStage[S, R, D]) fromChan(src <-chan S) {
	for {
		s.flush()
		if !s.pulling() {
			return
		}
		var in <-chan S
		if s.held < s.limit {
			in = src
		}
		select {
		case v, ok := <-in:
			if !ok {
				return
			}
			s.spawn(v)
		case <-s.wake:
		case <-s.gdone:
			return
		}
	}
}

func (s *groupStage[S, R, D]) slot() bool {
	for {
		s.flush()
		if !s.pulling() {
			return false
		}
		if s.held < s.limit {
			return true
		}
		s.await()
	}
}

func (s *groupStage[S, R, D]) drain() {
	for {
		s.flush()
		if s.held == 0 || !s.live() {
			return
		}
		s.await()
	}
}

func (s *groupStage[S, R, D]) await() {
	select {
	case <-s.wake:
	case <-s.gdone:
	}
}

func (s *groupStage[S, R, D]) spawn(v S) {
	idx := s.spawned
	s.spawned++
	s.held++
	if !s.enqueue(groupJob[S]{idx: idx, v: v}) && s.started < s.limit {
		s.started++
		s.g.Go(s.work)
	}
}

func (s *groupStage[S, R, D]) enqueue(j groupJob[S]) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.jobs) == cap(s.jobs) && s.jhead > 0 {
		n := copy(s.jobs, s.jobs[s.jhead:])
		clear(s.jobs[n:])
		s.jobs, s.jhead = s.jobs[:n], 0
	}
	s.jobs = append(s.jobs, j)
	if s.idle == 0 {
		return false
	}
	s.idle--
	s.cond.Signal()
	return true
}

func (s *groupStage[S, R, D]) work(ctx context.Context) error {
	for {
		j, ok := s.take()
		if !ok {
			return nil
		}
		it := s.call(ctx, j)
		if it.err != nil && s.completion {
			return it.err
		}
		s.finish(it)
	}
}

func (s *groupStage[S, R, D]) call(ctx context.Context, j groupJob[S]) (it groupItem[R]) {
	it.idx = j.idx
	if !s.completion {
		defer func() {
			if p := recover(); p != nil {
				it.err = &PanicError{Value: p, Stack: stackTrace()}
			}
		}()
	}
	it.r, it.keep, it.err = s.f(ctx, j.v)
	if it.err != nil && errors.Is(it.err, ErrNotOwned) {
		it.err, it.unowned = nil, true
	}
	return it
}

func (s *groupStage[S, R, D]) take() (groupJob[S], bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for s.jhead == len(s.jobs) && !s.closed {
		s.idle++
		s.cond.Wait()
	}
	if s.jhead == len(s.jobs) {
		return groupJob[S]{}, false
	}
	j := s.jobs[s.jhead]
	s.jobs[s.jhead] = groupJob[S]{}
	s.jhead++
	if s.jhead == len(s.jobs) {
		s.jobs, s.jhead = s.jobs[:0], 0
	}
	return j, true
}

func (s *groupStage[S, R, D]) stopWorkers() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	s.cond.Broadcast()
}

func (s *groupStage[S, R, D]) finish(it groupItem[R]) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if it.err != nil {
		s.failKnown.Store(true)
	}
	key := it.idx
	if s.completion {
		key = s.done
	}
	s.done++
	s.buf[key] = it
	select {
	case s.wake <- struct{}{}:
	default:
	}
}

func (s *groupStage[S, R, D]) pop() (groupItem[R], bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	it, ok := s.buf[s.next]
	if ok {
		delete(s.buf, s.next)
		s.next++
	}
	return it, ok
}

func (s *groupStage[S, R, D]) flush() {
	for s.live() {
		it, ok := s.pop()
		if !ok {
			return
		}
		s.process(it)
	}
}

func (s *groupStage[S, R, D]) process(it groupItem[R]) {
	s.held--
	if it.err != nil {
		s.headErr = it.err
		return
	}
	if !it.keep {
		if !it.unowned {
			s.discard(it.r)
		}
		return
	}
	if !s.yield(it.r, nil) {
		s.stopped = true
	}
}

func (s *groupStage[S, R, D]) conclude(ctx context.Context) {
	err := s.join()
	if s.stopped {
		return
	}
	err = cmp.Or(s.headErr, err, ctx.Err())
	if err != nil {
		var zero R
		s.yield(zero, err)
	}
}

func (s *groupStage[S, R, D]) join() error {
	if !s.joined {
		s.joined = true
		s.cancel()
		s.stopWorkers()
		s.err = s.g.Wait()
		s.abandon()
	}
	return s.err
}

func (s *groupStage[S, R, D]) abandon() {
	lost := make([]groupItem[R], 0, len(s.buf))
	for _, it := range s.buf {
		if it.err == nil && !it.unowned {
			lost = append(lost, it)
		}
	}
	slices.SortFunc(lost, func(a, b groupItem[R]) int { return cmp.Compare(b.idx, a.idx) })
	s.discardAll(lost)
}

func (s *groupStage[S, R, D]) discardAll(items []groupItem[R]) {
	if len(items) == 0 {
		return
	}
	defer s.discardAll(items[1:])
	s.discard(items[0].r)
}

func (s *groupStage[S, R, D]) discard(r R) {
	var d D
	d.Discard(r)
}
