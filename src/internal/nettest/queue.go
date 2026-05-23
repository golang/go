// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nettest

type queue[T any] struct {
	headPos int
	head    []T
	tail    []T
}

func (q *queue[T]) len() int {
	return len(q.head[q.headPos:]) + len(q.tail)
}

func (q *queue[T]) push(v T) {
	q.tail = append(q.tail, v)
}

func (q *queue[T]) pop() T {
	var zero T
	if q.headPos >= len(q.head) {
		if len(q.tail) == 0 {
			return zero
		}
		q.head, q.headPos, q.tail = q.tail, 0, q.head[:0]
	}
	v := q.head[q.headPos]
	q.head[q.headPos] = zero
	q.headPos++
	return v
}
