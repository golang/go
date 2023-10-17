// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package par

import "fmt"

// Queue manages a set of work items to be executed in parallel. The number of
// active work items is limited, and excess items are queued sequentially.
type Queue struct {
	maxActive int
	st        chan queueState
}

type queueState struct {
	active  int // number of goroutines processing work; always nonzero when len(backlog) > 0
	backlog []func()
	idle    chan struct{} // if non-nil, closed when active becomes 0
}

// NewQueue returns a Queue that executes up to maxActive items in parallel.
//
// maxActive must be positive.
func NewQueue(maxActive int) *Queue {
	if maxActive < 1 {
		panic(fmt.Sprintf("par.NewQueue called with nonpositive limit (%d)", maxActive))
	}

	q := &Queue{
		maxActive: maxActive,
		st:        make(chan queueState, 1),
	}
	q.st <- queueState{}
	return q
}

// Add adds f as a work item in the queue.
//
// Add returns immediately, but the queue will be marked as non-idle until after
// f (and any subsequently-added work) has completed.
func (q *Queue) Add(f func()) {
	st := <-q.st
	if st.active == q.maxActive {
		st.backlog = append(st.backlog, f)
		q.st <- st
		return
	}
	if st.active == 0 {
		// Mark q as non-idle.
		st.idle = nil
	}
	st.active++
	q.st <- st

	go func() {
		for {
			f()

			st := <-q.st
			if len(st.backlog) == 0 {
				if st.active--; st.active == 0 && st.idle != nil {
					close(st.idle)
				}
				q.st <- st
				return
			}
			f, st.backlog = st.backlog[0], st.backlog[1:]
			q.st <- st
		}
	}()
}

// Idle returns a channel that will be closed when q has no (active or enqueued)
// work outstanding.
func (q *Queue) Idle() <-chan struct{} {
	st := <-q.st
	defer func() { q.st <- st }()

	if st.idle == nil {
		st.idle = make(chan struct{})
		if st.active == 0 {
			close(st.idle)
		}
	}

	return st.idle
}
