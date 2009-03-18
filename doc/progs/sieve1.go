// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

// Send the sequence 2, 3, 4, ... to returned channel 
func generate() chan int {
	ch := make(chan int);
	go func(ch chan int){
		for i := 2; ; i++ {
			ch <- i
		}
	}(ch);
	return ch;
}

// Filter out input values divisible by 'prime', send rest to returned channel
func filter(in chan int, prime int) chan int {
	out := make(chan int);
	go func(in chan int, out chan int, prime int) {
		for {
			if i := <-in; i % prime != 0 {
				out <- i
			}
		}
	}(in, out, prime);
	return out;
}

func sieve() chan int {
	out := make(chan int);
	go func(out chan int) {
		ch := generate();
		for {
			prime := <-ch;
			out <- prime;
			ch = filter(ch, prime);
		}
	}(out);
	return out;
}

func main() {
	primes := sieve();
	for {
		fmt.Println(<-primes);
	}
}
