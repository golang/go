// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.


package main

import big "gmp"
//import "big"
import "fmt"

func Fib(n int) *big.Int {
	a := big.NewInt(0);
	b := big.NewInt(1);

	for i := 0; i < n; i++ {
		a, b = b, a;
		b.Add(a, b);
	}

	return b;
}

func main() {
	for i := 0; i <= 100; i++ {
		fmt.Println(5*i, Fib(5*i));
	}
}
