// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package goerror_fp

type Seq[T any] []T

func (r Seq[T]) Size() int {
	return len(r)
}

func (r Seq[T]) Append(items ...T) Seq[T] {
	tail := Seq[T](items)
	ret := make(Seq[T], r.Size()+tail.Size())

	for i := range r {
		ret[i] = r[i]
	}

	for i := range tail {
		ret[i+r.Size()] = tail[i]
	}

	return ret
}

func (r Seq[T]) Iterator() Iterator[T] {
	idx := 0

	return Iterator[T]{
		IsHasNext: func() bool {
			return idx < r.Size()
		},
		GetNext: func() T {
			ret := r[idx]
			idx++
			return ret
		},
	}
}

type Iterator[T any] struct {
	IsHasNext func() bool
	GetNext   func() T
}

func (r Iterator[T]) ToSeq() Seq[T] {
	ret := Seq[T]{}
	for r.HasNext() {
		ret = append(ret, r.Next())
	}
	return ret
}

func (r Iterator[T]) Map(f func(T) any) Iterator[any] {
	return MakeIterator(r.HasNext, func() any {
		return f(r.Next())
	})
}

func (r Iterator[T]) HasNext() bool {
	return r.IsHasNext()
}

func (r Iterator[T]) Next() T {
	return r.GetNext()
}

func MakeIterator[T any](has func() bool, next func() T) Iterator[T] {
	return Iterator[T]{
		IsHasNext: has,
		GetNext:   next,
	}
}
