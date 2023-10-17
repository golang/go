// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

// _List is a linked list of ordered values of type T.
type _List[T Ordered] struct {
	next *_List[T]
	val  T
}

func (l *_List[T]) Largest() T {
	var max T
	for p := l; p != nil; p = p.next {
		if p.val > max {
			max = p.val
		}
	}
	return max
}

type OrderedNum interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// _ListNum is a linked _List of ordered numeric values of type T.
type _ListNum[T OrderedNum] struct {
	next *_ListNum[T]
	val  T
}

const Clip = 5

// ClippedLargest returns the largest in the list of OrderNums, but a max of 5.
// Test use of untyped constant in an expression with a generically-typed parameter
func (l *_ListNum[T]) ClippedLargest() T {
	var max T
	for p := l; p != nil; p = p.next {
		if p.val > max && p.val < Clip {
			max = p.val
		}
	}
	return max
}

func main() {
	i3 := &_List[int]{nil, 1}
	i2 := &_List[int]{i3, 3}
	i1 := &_List[int]{i2, 2}
	if got, want := i1.Largest(), 3; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	b3 := &_List[byte]{nil, byte(1)}
	b2 := &_List[byte]{b3, byte(3)}
	b1 := &_List[byte]{b2, byte(2)}
	if got, want := b1.Largest(), byte(3); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	f3 := &_List[float64]{nil, 13.5}
	f2 := &_List[float64]{f3, 1.2}
	f1 := &_List[float64]{f2, 4.5}
	if got, want := f1.Largest(), 13.5; got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}

	s3 := &_List[string]{nil, "dd"}
	s2 := &_List[string]{s3, "aa"}
	s1 := &_List[string]{s2, "bb"}
	if got, want := s1.Largest(), "dd"; got != want {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}

	j3 := &_ListNum[int]{nil, 1}
	j2 := &_ListNum[int]{j3, 32}
	j1 := &_ListNum[int]{j2, 2}
	if got, want := j1.ClippedLargest(), 2; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	g3 := &_ListNum[float64]{nil, 13.5}
	g2 := &_ListNum[float64]{g3, 1.2}
	g1 := &_ListNum[float64]{g2, 4.5}
	if got, want := g1.ClippedLargest(), 4.5; got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}
}
