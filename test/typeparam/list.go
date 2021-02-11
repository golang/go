// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type Ordered interface {
        type int, int8, int16, int32, int64,
                uint, uint8, uint16, uint32, uint64, uintptr,
                float32, float64,
                string
}

// List is a linked list of ordered values of type T.
type list[T Ordered] struct {
	next *list[T]
	val  T
}

func (l *list[T]) largest() T {
	var max T
	for p := l; p != nil; p = p.next {
		if p.val > max {
			max = p.val
		}
	}
	return max
}


func main() {
	i3 := &list[int]{nil, 1}
	i2 := &list[int]{i3, 3}
	i1 := &list[int]{i2, 2}
	if got, want := i1.largest(), 3; got != want {
                panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	b3 := &list[byte]{nil, byte(1)}
	b2 := &list[byte]{b3, byte(3)}
	b1 := &list[byte]{b2, byte(2)}
	if got, want := b1.largest(), byte(3); got != want {
                panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	f3 := &list[float64]{nil, 13.5}
	f2 := &list[float64]{f3, 1.2}
	f1 := &list[float64]{f2, 4.5}
	if got, want := f1.largest(), 13.5; got != want {
                panic(fmt.Sprintf("got %f, want %f", got, want))
	}

	s3 := &list[string]{nil, "dd"}
	s2 := &list[string]{s3, "aa"}
	s1 := &list[string]{s2, "bb"}
	if got, want := s1.largest(), "dd"; got != want {
                panic(fmt.Sprintf("got %s, want %s", got, want))
	}
}
