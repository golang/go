// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Generate primes up to 100 using channels, checking the results.
// This sieve is Eratosthenesque and only considers odd candidates.
// See discussion at <http://blog.onideas.ws/eratosthenes.go>.

package main

import (
	"container/heap"
	"container/ring"
	"container/vector"
)

// Return a chan of odd numbers, starting from 5.
func odds() chan int {
	out := make(chan int, 50)
	go func() {
		n := 5
		for {
			out <- n
			n += 2
		}
	}()
	return out
}

// Return a chan of odd multiples of the prime number p, starting from p*p.
func multiples(p int) chan int {
	out := make(chan int, 10)
	go func() {
		n := p * p
		for {
			out <- n
			n += 2 * p
		}
	}()
	return out
}

type PeekCh struct {
	head int
	ch   chan int
}

// Heap of PeekCh, sorting by head values.
type PeekChHeap struct {
	*vector.Vector
}

func (h *PeekChHeap) Less(i, j int) bool {
	return h.At(i).(*PeekCh).head < h.At(j).(*PeekCh).head
}

// Return a channel to serve as a sending proxy to 'out'.
// Use a goroutine to receive values from 'out' and store them
// in an expanding buffer, so that sending to 'out' never blocks.
func sendproxy(out chan<- int) chan<- int {
	proxy := make(chan int, 10)
	go func() {
		n := 16 // the allocated size of the circular queue
		first := ring.New(n)
		last := first
		var c chan<- int
		var e int
		for {
			c = out
			if first == last {
				// buffer empty: disable output
				c = nil
			} else {
				e = first.Value.(int)
			}
			select {
			case e = <-proxy:
				last.Value = e
				if last.Next() == first {
					// buffer full: expand it
					last.Link(ring.New(n))
					n *= 2
				}
				last = last.Next()
			case c <- e:
				first = first.Next()
			}
		}
	}()
	return proxy
}

// Return a chan int of primes.
func Sieve() chan int {
	// The output values.
	out := make(chan int, 10)
	out <- 2
	out <- 3

	// The channel of all composites to be eliminated in increasing order.
	composites := make(chan int, 50)

	// The feedback loop.
	primes := make(chan int, 10)
	primes <- 3

	// Merge channels of multiples of 'primes' into 'composites'.
	go func() {
		h := &PeekChHeap{new(vector.Vector)}
		min := 15
		for {
			m := multiples(<-primes)
			head := <-m
			for min < head {
				composites <- min
				minchan := heap.Pop(h).(*PeekCh)
				min = minchan.head
				minchan.head = <-minchan.ch
				heap.Push(h, minchan)
			}
			for min == head {
				minchan := heap.Pop(h).(*PeekCh)
				min = minchan.head
				minchan.head = <-minchan.ch
				heap.Push(h, minchan)
			}
			composites <- head
			heap.Push(h, &PeekCh{<-m, m})
		}
	}()

	// Sieve out 'composites' from 'candidates'.
	go func() {
		// In order to generate the nth prime we only need multiples of
		// primes ≤ sqrt(nth prime).  Thus, the merging goroutine will
		// receive from 'primes' much slower than this goroutine
		// will send to it, making the buffer accumulate and block this
		// goroutine from sending, causing a deadlock.  The solution is to
		// use a proxy goroutine to do automatic buffering.
		primes := sendproxy(primes)

		candidates := odds()
		p := <-candidates

		for {
			c := <-composites
			for p < c {
				primes <- p
				out <- p
				p = <-candidates
			}
			if p == c {
				p = <-candidates
			}
		}
	}()

	return out
}

func main() {
	primes := Sieve()
	a := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
	for i := 0; i < len(a); i++ {
		if x := <-primes; x != a[i] {
			println(x, " != ", a[i])
			panic("fail")
		}
	}
}
