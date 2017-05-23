// skip

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Usage:
// fibo <n>     compute fibonacci(n), n must be >= 0
// fibo -bench  benchmark fibonacci computation (takes about 1 min)
//
// Additional flags:
// -half        add values using two half-digit additions
// -opt         optimize memory allocation through reuse
// -short       only print the first 10 digits of very large fibonacci numbers

// Command fibo is a stand-alone test and benchmark to
// evaluate the performance of bignum arithmetic written
// entirely in Go.
package main

import (
	"flag"
	"fmt"
	"math/big" // only used for printing
	"os"
	"strconv"
	"testing"
	"text/tabwriter"
	"time"
)

var (
	bench = flag.Bool("bench", false, "run benchmarks")
	half  = flag.Bool("half", false, "use half-digit addition")
	opt   = flag.Bool("opt", false, "optimize memory usage")
	short = flag.Bool("short", false, "only print first 10 digits of result")
)

// A large natural number is represented by a nat, each "digit" is
// a big.Word; the value zero corresponds to the empty nat slice.
type nat []big.Word

const W = 1 << (5 + ^big.Word(0)>>63) // big.Word size in bits

// The following methods are extracted from math/big to make this a
// stand-alone program that can easily be run without dependencies
// and compiled with different compilers.

func (z nat) make(n int) nat {
	if n <= cap(z) {
		return z[:n] // reuse z
	}
	// Choosing a good value for e has significant performance impact
	// because it increases the chance that a value can be reused.
	const e = 4 // extra capacity
	return make(nat, n, n+e)
}

// z = x
func (z nat) set(x nat) nat {
	z = z.make(len(x))
	copy(z, x)
	return z
}

// z = x + y
// (like add, but operating on half-digits at a time)
func (z nat) halfAdd(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.add(y, x)
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z.make(0)
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m >= n > 0

	const W2 = W / 2         // half-digit size in bits
	const M2 = (1 << W2) - 1 // lower half-digit mask

	z = z.make(m + 1)
	var c big.Word
	for i := 0; i < n; i++ {
		// lower half-digit
		c += x[i]&M2 + y[i]&M2
		d := c & M2
		c >>= W2
		// upper half-digit
		c += x[i]>>W2 + y[i]>>W2
		z[i] = c<<W2 | d
		c >>= W2
	}
	for i := n; i < m; i++ {
		// lower half-digit
		c += x[i] & M2
		d := c & M2
		c >>= W2
		// upper half-digit
		c += x[i] >> W2
		z[i] = c<<W2 | d
		c >>= W2
	}
	if c != 0 {
		z[m] = c
		m++
	}
	return z[:m]
}

// z = x + y
func (z nat) add(x, y nat) nat {
	m := len(x)
	n := len(y)

	switch {
	case m < n:
		return z.add(y, x)
	case m == 0:
		// n == 0 because m >= n; result is 0
		return z.make(0)
	case n == 0:
		// result is x
		return z.set(x)
	}
	// m >= n > 0

	z = z.make(m + 1)
	var c big.Word

	for i, xi := range x[:n] {
		yi := y[i]
		zi := xi + yi + c
		z[i] = zi
		// see "Hacker's Delight", section 2-12 (overflow detection)
		c = ((xi & yi) | ((xi | yi) &^ zi)) >> (W - 1)
	}
	for i, xi := range x[n:] {
		zi := xi + c
		z[n+i] = zi
		c = (xi &^ zi) >> (W - 1)
		if c == 0 {
			copy(z[n+i+1:], x[i+1:])
			break
		}
	}
	if c != 0 {
		z[m] = c
		m++
	}
	return z[:m]
}

func bitlen(x big.Word) int {
	n := 0
	for x > 0 {
		x >>= 1
		n++
	}
	return n
}

func (x nat) bitlen() int {
	if i := len(x); i > 0 {
		return (i-1)*W + bitlen(x[i-1])
	}
	return 0
}

func (x nat) String() string {
	const shortLen = 10
	s := new(big.Int).SetBits(x).String()
	if *short && len(s) > shortLen {
		s = s[:shortLen] + "..."
	}
	return s
}

func fibo(n int, half, opt bool) nat {
	switch n {
	case 0:
		return nil
	case 1:
		return nat{1}
	}
	f0 := nat(nil)
	f1 := nat{1}
	if half {
		if opt {
			var f2 nat // reuse f2
			for i := 1; i < n; i++ {
				f2 = f2.halfAdd(f1, f0)
				f0, f1, f2 = f1, f2, f0
			}
		} else {
			for i := 1; i < n; i++ {
				f2 := nat(nil).halfAdd(f1, f0) // allocate a new f2 each time
				f0, f1 = f1, f2
			}
		}
	} else {
		if opt {
			var f2 nat // reuse f2
			for i := 1; i < n; i++ {
				f2 = f2.add(f1, f0)
				f0, f1, f2 = f1, f2, f0
			}
		} else {
			for i := 1; i < n; i++ {
				f2 := nat(nil).add(f1, f0) // allocate a new f2 each time
				f0, f1 = f1, f2
			}
		}
	}
	return f1 // was f2 before shuffle
}

var tests = []struct {
	n    int
	want string
}{
	{0, "0"},
	{1, "1"},
	{2, "1"},
	{3, "2"},
	{4, "3"},
	{5, "5"},
	{6, "8"},
	{7, "13"},
	{8, "21"},
	{9, "34"},
	{10, "55"},
	{100, "354224848179261915075"},
	{1000, "43466557686937456435688527675040625802564660517371780402481729089536555417949051890403879840079255169295922593080322634775209689623239873322471161642996440906533187938298969649928516003704476137795166849228875"},
}

func test(half, opt bool) {
	for _, test := range tests {
		got := fibo(test.n, half, opt).String()
		if got != test.want {
			fmt.Printf("error: got std fibo(%d) = %s; want %s\n", test.n, got, test.want)
			os.Exit(1)
		}
	}
}

func selfTest() {
	if W != 32 && W != 64 {
		fmt.Printf("error: unexpected wordsize %d", W)
		os.Exit(1)
	}
	for i := 0; i < 4; i++ {
		test(i&2 == 0, i&1 != 0)
	}
}

func doFibo(n int) {
	start := time.Now()
	f := fibo(n, *half, *opt)
	t := time.Since(start)
	fmt.Printf("fibo(%d) = %s (%d bits, %s)\n", n, f, f.bitlen(), t)
}

func benchFibo(b *testing.B, n int, half, opt bool) {
	for i := 0; i < b.N; i++ {
		fibo(n, half, opt)
	}
}

func doBench(half, opt bool) {
	w := tabwriter.NewWriter(os.Stdout, 0, 8, 2, ' ', tabwriter.AlignRight)
	fmt.Fprintf(w, "wordsize = %d, half = %v, opt = %v\n", W, half, opt)
	fmt.Fprintf(w, "n\talloc count\talloc bytes\tns/op\ttime/op\t\n")
	for n := 1; n <= 1e6; n *= 10 {
		res := testing.Benchmark(func(b *testing.B) { benchFibo(b, n, half, opt) })
		fmt.Fprintf(w, "%d\t%d\t%d\t%d\t%s\t\n", n, res.AllocsPerOp(), res.AllocedBytesPerOp(), res.NsPerOp(), time.Duration(res.NsPerOp()))
	}
	fmt.Fprintln(w)
	w.Flush()
}

func main() {
	selfTest()
	flag.Parse()

	if args := flag.Args(); len(args) > 0 {
		// command-line use
		fmt.Printf("half = %v, opt = %v, wordsize = %d bits\n", *half, *opt, W)
		for _, arg := range args {
			n, err := strconv.Atoi(arg)
			if err != nil || n < 0 {
				fmt.Println("invalid argument", arg)
				continue
			}
			doFibo(n)
		}
		return
	}

	if *bench {
		for i := 0; i < 4; i++ {
			doBench(i&2 == 0, i&1 != 0)
		}
	}
}
