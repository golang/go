// $G $D/$F.go || echo BUG should compile

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test cases for conversion syntax.

package main

type (
	A [3]int
	S struct {
		x int
	}
	P *S
	F func(x int) int
	I interface {
		m(x int) int
	}
	L []int
	M map[string]int
	C chan int
)

func (s S) m(x int) int { return x }

var (
	a A = [...]int{1, 2, 3}
	s S = struct{ x int }{0}
	p P = &s
	f F = func(x int) int { return x }
	i I = s
	l L = []int{}
	m M = map[string]int{"foo": 0}
	c C = make(chan int)
)

func main() {
	a = A(a)
	a = [3]int(a)
	s = struct {
		x int
	}(s)
	p = (*S)(p)
	f = func(x int) int(f)
	i = (interface {
		m(x int) int
	})(s) // this is accepted by 6g
	i = interface {
		m(x int) int
	}(s) // this is not accepted by 6g (but should be)
	l = []int(l)
	m = map[string]int(m)
	c = chan int(c)
	_ = chan<- int(c)
	_ = <-(chan int)(c)
	_ = <-(<-chan int)(c)
}

/*
6g bug277.go
bug277.go:46: syntax error: unexpected (, expecting {
bug277.go:50: syntax error: unexpected interface
bug277.go:53: non-declaration statement outside function body
bug277.go:54: non-declaration statement outside function body
bug277.go:55: syntax error: unexpected LCHAN
bug277.go:56: syntax error: unexpected LCHAN
bug277.go:57: non-declaration statement outside function body
bug277.go:58: non-declaration statement outside function body
bug277.go:59: syntax error: unexpected }
*/
