// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzz

// queue holds a growable sequence of inputs for fuzzing and minimization.
//
// For now, this is a simple ring buffer
// (https://en.wikipedia.org/wiki/Circular_buffer).
//
// TODO(golang.org/issue/46224): use a prioritization algorithm based on input
// size, previous duration, coverage, and any other metrics that seem useful.
type queue struct {
	// elems holds a ring buffer.
	// The queue is empty when begin = end.
	// The queue is full (until grow is called) when end = begin + N - 1 (mod N)
	// where N = cap(elems).
	elems     []any
	head, len int
}

func (q *queue) cap() int {
	return len(q.elems)
}

func (q *queue) grow() {
	oldCap := q.cap()
	newCap := oldCap * 2
	if newCap == 0 {
		newCap = 8
	}
	newElems := make([]any, newCap)
	oldLen := q.len
	for i := 0; i < oldLen; i++ {
		newElems[i] = q.elems[(q.head+i)%oldCap]
	}
	q.elems = newElems
	q.head = 0
}

func (q *queue) enqueue(e any) {
	if q.len+1 > q.cap() {
		q.grow()
	}
	i := (q.head + q.len) % q.cap()
	q.elems[i] = e
	q.len++
}

func (q *queue) dequeue() (any, bool) {
	if q.len == 0 {
		return nil, false
	}
	e := q.elems[q.head]
	q.elems[q.head] = nil
	q.head = (q.head + 1) % q.cap()
	q.len--
	return e, true
}

func (q *queue) peek() (any, bool) {
	if q.len == 0 {
		return nil, false
	}
	return q.elems[q.head], true
}

func (q *queue) clear() {
	*q = queue{}
}
