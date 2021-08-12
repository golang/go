// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

// List is a linked list of ordered values of type T.
type List[T Ordered] struct {
	Next *List[T]
	Val  T
}

func (l *List[T]) Largest() T {
	var max T
	for p := l; p != nil; p = p.Next {
		if p.Val > max {
			max = p.Val
		}
	}
	return max
}

type OrderedNum interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// ListNum is a linked _List of ordered numeric values of type T.
type ListNum[T OrderedNum] struct {
	Next *ListNum[T]
	Val  T
}

const Clip = 5

// clippedLargest returns the largest in the list of OrderNums, but a max of 5.
func (l *ListNum[T]) ClippedLargest() T {
	var max T
	for p := l; p != nil; p = p.Next {
		if p.Val > max && p.Val < Clip {
			max = p.Val
		}
	}
	return max
}
