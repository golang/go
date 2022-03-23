// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"./a"
	"fmt"
)

func main() {
	i3 := &a.List[int]{nil, 1}
	i2 := &a.List[int]{i3, 3}
	i1 := &a.List[int]{i2, 2}
	if got, want := i1.Largest(), 3; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	b3 := &a.List[byte]{nil, byte(1)}
	b2 := &a.List[byte]{b3, byte(3)}
	b1 := &a.List[byte]{b2, byte(2)}
	if got, want := b1.Largest(), byte(3); got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}

	f3 := &a.List[float64]{nil, 13.5}
	f2 := &a.List[float64]{f3, 1.2}
	f1 := &a.List[float64]{f2, 4.5}
	if got, want := f1.Largest(), 13.5; got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}

	s3 := &a.List[string]{nil, "dd"}
	s2 := &a.List[string]{s3, "aa"}
	s1 := &a.List[string]{s2, "bb"}
	if got, want := s1.Largest(), "dd"; got != want {
		panic(fmt.Sprintf("got %s, want %s", got, want))
	}
	j3 := &a.ListNum[int]{nil, 1}
	j2 := &a.ListNum[int]{j3, 32}
	j1 := &a.ListNum[int]{j2, 2}
	if got, want := j1.ClippedLargest(), 2; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
	g3 := &a.ListNum[float64]{nil, 13.5}
	g2 := &a.ListNum[float64]{g3, 1.2}
	g1 := &a.ListNum[float64]{g2, 4.5}
	if got, want := g1.ClippedLargest(), 4.5; got != want {
		panic(fmt.Sprintf("got %f, want %f", got, want))
	}
}
