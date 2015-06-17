// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"log"
	"math/big"
)

func ExampleRat_SetString() {
	r := new(big.Rat)
	r.SetString("355/113")
	fmt.Println(r.FloatString(3))
	// Output: 3.142
}

func ExampleInt_SetString() {
	i := new(big.Int)
	i.SetString("644", 8) // octal
	fmt.Println(i)
	// Output: 420
}

func ExampleRat_Scan() {
	// The Scan function is rarely used directly;
	// the fmt package recognizes it as an implementation of fmt.Scanner.
	r := new(big.Rat)
	_, err := fmt.Sscan("1.5000", r)
	if err != nil {
		log.Println("error scanning value:", err)
	} else {
		fmt.Println(r)
	}
	// Output: 3/2
}

func ExampleInt_Scan() {
	// The Scan function is rarely used directly;
	// the fmt package recognizes it as an implementation of fmt.Scanner.
	i := new(big.Int)
	_, err := fmt.Sscan("18446744073709551617", i)
	if err != nil {
		log.Println("error scanning value:", err)
	} else {
		fmt.Println(i)
	}
	// Output: 18446744073709551617
}

// Example_fibonacci demonstrates how to use big.Int to compute the smallest
// Fibonacci number with 100 decimal digits, and find out whether it is prime.
func Example_fibonacci() {
	// create and initialize big.Ints from int64s
	fib1 := big.NewInt(0)
	fib2 := big.NewInt(1)

	// initialize limit as 10^99 (the smallest integer with 100 digits)
	var limit big.Int
	limit.Exp(big.NewInt(10), big.NewInt(99), nil)

	// loop while fib1 is smaller than 1e100
	for fib1.Cmp(&limit) < 0 {
		fib1, fib2 = fib2, fib1.Add(fib1, fib2)
	}

	fmt.Println(fib1) // 100-digits fibonacci number

	// Test fib1 for primality. The ProbablyPrimes parameter sets the number
	// of Miller-Rabin rounds to be performed. 20 is a good value.
	isPrime := fib1.ProbablyPrime(20)
	fmt.Println(isPrime) // false

	// Output:
	// 1344719667586153181419716641724567886890850696275767987106294472017884974410332069524504824747437757
	// false
}
