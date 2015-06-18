// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package big_test

import (
	"fmt"
	"log"
	"math"
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
		// Compute the next Fibonacci number:
		//    t1 := fib2
		//    t2 := fib1.Add(fib1, fib2) // Note that Add "assigns" to fib1!
		//    fib1 = t1
		//    fib2 = t2
		// Using Go's multi-value ("parallel") assignment, we can simply write:
		fib1, fib2 = fib2, fib1.Add(fib1, fib2)
	}

	fmt.Println(fib1) // 100-digit Fibonacci number

	// Test fib1 for primality. The ProbablyPrimes parameter sets the number
	// of Miller-Rabin rounds to be performed. 20 is a good value.
	isPrime := fib1.ProbablyPrime(20)
	fmt.Println(isPrime)

	// Output:
	// 1344719667586153181419716641724567886890850696275767987106294472017884974410332069524504824747437757
	// false
}

// This example shows how to use big.Float to compute the square root of 2 with
// a precision of 200 bits, and how to print the result as a decimal number.
func Example_sqrt2() {
	// We'll do computations with 200 bits of precision in the mantissa.
	const prec = 200

	// Compute the square root of 2 using Newton's Method. We start with
	// an initial estimate for sqrt(2), and then iterate:
	//     x_{n+1} = 1/2 * ( x_n + (2.0 / x_n) )

	// Since Newton's Method doubles the number of correct digits at each
	// iteration, we need at least log_2(prec) steps.
	steps := int(math.Log2(prec))

	// Initialize values we need for the computation.
	two := new(big.Float).SetPrec(prec).SetInt64(2)
	half := new(big.Float).SetPrec(prec).SetFloat64(0.5)

	// Use 1 as the initial estimate.
	x := new(big.Float).SetPrec(prec).SetInt64(1)

	// We use t as a temporary variable. There's no need to set its precision
	// since big.Float values with unset (== 0) precision automatically assume
	// the largest precision of the arguments when used as the result (receiver)
	// of a big.Float operation.
	t := new(big.Float)

	// Iterate.
	for i := 0; i <= steps; i++ {
		t.Quo(two, x)  // t = 2.0 / x_n
		t.Add(x, t)    // t = x_n + (2.0 / x_n)
		x.Mul(half, t) // x_{n+1} = 0.5 * t
	}

	// We can use the usual fmt.Printf verbs since big.Float implements fmt.Formatter
	fmt.Printf("sqrt(2) = %.50f\n", x)

	// Print the error between 2 and x*x.
	t.Mul(x, x) // t = x*x
	fmt.Printf("error = %e\n", t.Sub(two, t))

	// Output:
	// sqrt(2) = 1.41421356237309504880168872420969807856967187537695
	// error = 0.000000e+00
}
