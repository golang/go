// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This version generates up to 100 and checks the results.
// With a channel, of course.

package main

// Send the sequence 2, 3, 4, ... to channel 'ch'.
func Generate(ch *chan-< int) {
	for i := 2; ; i++ {
		ch -< i  // Send 'i' to channel 'ch'.
	}
}

// Copy the values from channel 'in' to channel 'out',
// removing those divisible by 'prime'.
func Filter(in *chan<- int, out *chan-< int, prime int) {
	for {
		i := <-in  // Receive value of new variable 'i' from 'in'.
		if i % prime != 0 {
			out -< i  // Send 'i' to channel 'out'.
		}
	}
}

// The prime sieve: Daisy-chain Filter processes together.
func Sieve(primes *chan-< int) {
	ch := new(chan int);  // Create a new channel.
	go Generate(ch);  // Start Generate() as a subprocess.
	for {
		prime := <-ch;
		primes -< prime;
		ch1 := new(chan int);
		go Filter(ch, ch1, prime);
		ch = ch1
	}
}

func main() {
	primes := new(chan int);
	go Sieve(primes);
	if <-primes != 2 { panic 2 }
	if <-primes != 3 { panic 3 }
	if <-primes != 5 { panic 5 }
	if <-primes != 7 { panic 7 }
	if <-primes != 11 { panic 11 }
	if <-primes != 13 { panic 13 }
	if <-primes != 17 { panic 17 }
	if <-primes != 19 { panic 19 }
	if <-primes != 23 { panic 23 }
	if <-primes != 29 { panic 29 }
	if <-primes != 31 { panic 31 }
	if <-primes != 37 { panic 37 }
	if <-primes != 41 { panic 41 }
	if <-primes != 43 { panic 43 }
	if <-primes != 47 { panic 47 }
	if <-primes != 53 { panic 53 }
	if <-primes != 59 { panic 59 }
	if <-primes != 61 { panic 61 }
	if <-primes != 67 { panic 67 }
	if <-primes != 71 { panic 71 }
	if <-primes != 73 { panic 73 }
	if <-primes != 79 { panic 79 }
	if <-primes != 83 { panic 83 }
	if <-primes != 89 { panic 89 }
	if <-primes != 97 { panic 97 }
	sys.exit(0);
}
