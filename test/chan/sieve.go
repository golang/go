// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This version generates up to 100 and checks the results.
// With a channel, of course.

package main

// Send the sequence 2, 3, 4, ... to channel 'ch'.
export func Generate(ch chan<- int) {
	for i := 2; ; i++ {
		ch <- i  // Send 'i' to channel 'ch'.
	}
}

// Copy the values from channel 'in' to channel 'out',
// removing those divisible by 'prime'.
export func Filter(in <-chan int, out chan<- int, prime int) {
	for {
		i := <-in;  // Receive value of new variable 'i' from 'in'.
		if i % prime != 0 {
			out <- i  // Send 'i' to channel 'out'.
		}
	}
}

// The prime sieve: Daisy-chain Filter processes together.
export func Sieve(primes chan<- int) {
	ch := make(chan int);  // Create a new channel.
	go Generate(ch);  // Start Generate() as a subprocess.
	for {
		prime := <-ch;
		primes <- prime;
		ch1 := make(chan int);
		go Filter(ch, ch1, prime);
		ch = ch1
	}
}

func main() {
	primes := make(chan int);
	go Sieve(primes);
	a := []int{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97};
	for i := 0; i < len(a); i++ {
		if <-primes != a[i] { panic(a[i])}
	}
	sys.Exit(0);
}
