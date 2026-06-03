// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import "context"

// A queue is an unbounded queue of some item (new connections and streams).
type queue[T any] struct {
	// The gate condition is set if the queue is non-empty or closed.
	gate gate
	err  error
	q    []T
}

func newQueue[T any]() queue[T] {
	return queue[T]{gate: newGate()}
}

// close closes the queue, causing pending and future pop operations
// to return immediately with err.
func (q *queue[T]) close(err error) {
	q.gate.lock()
	defer q.unlock()
	if q.err == nil {
		q.err = err
	}
}

// put appends an item to the queue.
// It returns true if the item was added, false if the queue is closed.
func (q *queue[T]) put(v T) bool {
	q.gate.lock()
	defer q.unlock()
	if q.err != nil {
		return false
	}
	q.q = append(q.q, v)
	return true
}

// get removes the first item from the queue, blocking until ctx is done, an item is available,
// or the queue is closed.
func (q *queue[T]) get(ctx context.Context) (T, error) {
	var zero T
	if err := q.gate.waitAndLock(ctx); err != nil {
		return zero, err
	}
	defer q.unlock()
	if q.err != nil {
		return zero, q.err
	}
	v := q.q[0]
	copy(q.q[:], q.q[1:])
	q.q[len(q.q)-1] = zero
	q.q = q.q[:len(q.q)-1]
	return v, nil
}

func (q *queue[T]) unlock() {
	q.gate.unlock(q.err != nil || len(q.q) > 0)
}
