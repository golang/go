// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

type payload struct {
	a, b, c int64
}

type node struct {
	next *node
	v    int64
}

var sinkHold any
var sinkIface any
var sinkApp any

var sizes = []int{32, 33, 255, 256, 257, 4095, 4096, 4097, 64511, 64512, 64513}

func enc[T any](i int) T {
	var v T
	switch p := any(&v).(type) {
	case *byte:
		*p = byte(i%251 + 1)
	case *int64:
		*p = int64(i)*3 + 7
	case *payload:
		*p = payload{int64(i), int64(i) + 1, int64(i) + 2}
	case *node:
		*p = node{v: int64(i)}
	default:
		panic("unexpected type")
	}
	return v
}

func cenc[T any](i int) T {
	var v T
	switch p := any(&v).(type) {
	case *byte:
		*p = 0x55
	case *int64:
		*p = 0x5555555555555555
	case *payload:
		*p = payload{85, 85, 85}
	case *node:
		*p = node{v: 85}
	default:
		panic("unexpected type")
	}
	return v
}

func fill[T any](s []T) {
	for i := range s {
		s[i] = enc[T](i)
	}
}

func verify[T comparable](s []T) bool {
	for i := range s {
		if s[i] != enc[T](i) {
			return false
		}
	}
	return true
}

func clobber[T any](n int) {
	b := make([]T, n)
	for i := range b {
		b[i] = cenc[T](i)
	}
	sinkHold = b
}

type box[T any] struct {
	buf []T
}

type stasher[T any] struct {
	s []T
}

func (x stasher[T]) stash() []T {
	return x.s
}

func route1[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	return func() []T { return s }
}

func route2[T any](n int) func() []T {
	b := box[T]{buf: make([]T, n)}
	fill(b.buf)
	return func() []T { return b.buf }
}

func route3[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	return func() []T { return s }
}

func route4[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	sinkIface = s
	return func() []T { return sinkIface.([]T) }
}

func route5[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	sinkApp = append([]T(nil), s...)
	return func() []T { return s }
}

func route6[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	h := stasher[T]{s}
	m := h.stash
	return func() []T { return m() }
}

func route7[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	defer func() { sinkHold = s }()
	clobber[T](n)
	return func() []T { return s }
}

func route8[T any](n int) func() []T {
	s := make([]T, n)
	fill(s)
	ch := make(chan []T, 1)
	go func() { ch <- s }()
	return func() []T { return <-ch }
}

func route9[T any](n int) func() T {
	s := make([]T, n)
	fill(s)
	sinkHold = s
	p := &s[0]
	return func() T { return *p }
}

func routeThunk[T any](r, n int) func() []T {
	switch r {
	case 1:
		return route1[T](n)
	case 2:
		return route2[T](n)
	case 3:
		return route3[T](n)
	case 4:
		return route4[T](n)
	case 5:
		return route5[T](n)
	case 6:
		return route6[T](n)
	case 7:
		return route7[T](n)
	case 8:
		return route8[T](n)
	}
	panic("bad route")
}

func runT[T comparable](r, n int) bool {
	if r == 9 {
		f := route9[T](n)
		clobber[T](n)
		return f() == enc[T](0)
	}
	th := routeThunk[T](r, n)
	clobber[T](n)
	return verify(th())
}

func runOne(t, r, n int) bool {
	switch t {
	case 0:
		return runT[byte](r, n)
	case 1:
		return runT[int64](r, n)
	case 2:
		return runT[payload](r, n)
	default:
		return runT[node](r, n)
	}
}

func elemCount(t, size int) int {
	var div int
	switch t {
	case 0:
		div = 1
	case 1:
		div = 8
	case 2:
		div = 24
	default:
		div = 8
	}
	n := size / div
	if n < 1 {
		n = 1
	}
	return n
}

func main() {
	caseNo := 0
	bad := 0
	for r := 1; r <= 9; r++ {
		for si, size := range sizes {
			t := (r + si) % 4
			caseNo++
			n := elemCount(t, size)
			if runOne(t, r, n) {
				fmt.Printf("case %03d r%d t%d n%d ok\n", caseNo, r, t, n)
			} else {
				bad++
				fmt.Printf("case %03d r%d t%d n%d LAW clobbered\n", caseNo, r, t, n)
			}
		}
	}
	if bad > 0 {
		fmt.Printf("LAW arena_escape %d of %d cases clobbered\n", bad, caseNo)
		os.Exit(1)
	}
	fmt.Println("arena_escape ok")
}
