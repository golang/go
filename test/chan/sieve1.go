// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test concurrency primitives: classical inefficient concurrent prime sieve.

// Generate primes up to 100 using channels, checking the results.
// This sieve consists of a linear chain of divisibility filters,
// equivalent to trial-dividing each n by all primes p ≤ n.

package main

// Send the sequence 2, 3, 4, ... to channel 'ch'.
func Generate(ch chan<- int) {
	for i := 2; ; i++ {
		ch <- i // Send 'i' to channel 'ch'.
	}
}

// Copy the values from channel 'in' to channel 'out',
// removing those divisible by 'prime'.
func Filter(in <-chan int, out chan<- int, prime int) {
	for i := range in { // Loop over values received from 'in'.
		if i%prime != 0 {
			out <- i // Send 'i' to channel 'out'.
		}
	}
}

// The prime sieve: Daisy-chain Filter processes together.
func Sieve(primes chan<- int) {
	ch := make(chan int) // Create a new channel.
	go Generate(ch)      // Start Generate() as a subprocess.
	for {
		// Note that ch is different on each iteration.
		prime := <-ch
		primes <- prime
		ch1 := make(chan int)
		go Filter(ch, ch1, prime)
		ch = ch1
	}
}

func main() {
	primes := make(chan int)
	go Sieve(primes)
	a := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97}
	for i := 0; i < len(a); i++ {
		if x := <-primes; x != a[i] {
			println(x, " != ", a[i])
			panic("fail")
		}
	}
}
