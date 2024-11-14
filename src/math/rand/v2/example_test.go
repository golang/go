// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
	"text/tabwriter"
	"time"
)

// These tests serve as an example but also make sure we don't change
// the output of the random number generator when given a fixed seed.

func Example() {
	answers := []string{
		"It is certain",
		"It is decidedly so",
		"Without a doubt",
		"Yes definitely",
		"You may rely on it",
		"As I see it yes",
		"Most likely",
		"Outlook good",
		"Yes",
		"Signs point to yes",
		"Reply hazy try again",
		"Ask again later",
		"Better not tell you now",
		"Cannot predict now",
		"Concentrate and ask again",
		"Don't count on it",
		"My reply is no",
		"My sources say no",
		"Outlook not so good",
		"Very doubtful",
	}
	fmt.Println("Magic 8-Ball says:", answers[rand.IntN(len(answers))])
}

// This example shows the use of each of the methods on a *Rand.
// The use of the global functions is the same, without the receiver.
func Example_rand() {
	// Create and seed the generator.
	// Typically a non-fixed seed should be used, such as Uint64(), Uint64().
	// Using a fixed seed will produce the same output on every run.
	r := rand.New(rand.NewPCG(1, 2))

	// The tabwriter here helps us generate aligned output.
	w := tabwriter.NewWriter(os.Stdout, 1, 1, 1, ' ', 0)
	defer w.Flush()
	show := func(name string, v1, v2, v3 any) {
		fmt.Fprintf(w, "%s\t%v\t%v\t%v\n", name, v1, v2, v3)
	}

	// Float32 and Float64 values are in [0, 1).
	show("Float32", r.Float32(), r.Float32(), r.Float32())
	show("Float64", r.Float64(), r.Float64(), r.Float64())

	// ExpFloat64 values have an average of 1 but decay exponentially.
	show("ExpFloat64", r.ExpFloat64(), r.ExpFloat64(), r.ExpFloat64())

	// NormFloat64 values have an average of 0 and a standard deviation of 1.
	show("NormFloat64", r.NormFloat64(), r.NormFloat64(), r.NormFloat64())

	// Int32, Int64, and Uint32 generate values of the given width.
	// The Int method (not shown) is like either Int32 or Int64
	// depending on the size of 'int'.
	show("Int32", r.Int32(), r.Int32(), r.Int32())
	show("Int64", r.Int64(), r.Int64(), r.Int64())
	show("Uint32", r.Uint32(), r.Uint32(), r.Uint32())

	// IntN, Int32N, and Int64N limit their output to be < n.
	// They do so more carefully than using r.Int()%n.
	show("IntN(10)", r.IntN(10), r.IntN(10), r.IntN(10))
	show("Int32N(10)", r.Int32N(10), r.Int32N(10), r.Int32N(10))
	show("Int64N(10)", r.Int64N(10), r.Int64N(10), r.Int64N(10))

	// Perm generates a random permutation of the numbers [0, n).
	show("Perm", r.Perm(5), r.Perm(5), r.Perm(5))
	// Output:
	// Float32     0.95955694          0.8076733            0.8135684
	// Float64     0.4297927436037299  0.797802349388613    0.3883664855410056
	// ExpFloat64  0.43463410545541104 0.5513632046504593   0.7426404617374481
	// NormFloat64 -0.9303318111676635 -0.04750789419852852 0.22248301107582735
	// Int32       2020777787          260808523            851126509
	// Int64       5231057920893523323 4257872588489500903  158397175702351138
	// Uint32      314478343           1418758728           208955345
	// IntN(10)    6                   2                    0
	// Int32N(10)  3                   7                    7
	// Int64N(10)  8                   9                    4
	// Perm        [0 3 1 4 2]         [4 1 2 0 3]          [4 3 2 0 1]
}

func ExamplePerm() {
	for _, value := range rand.Perm(3) {
		fmt.Println(value)
	}

	// Unordered output: 1
	// 2
	// 0
}

func ExampleN() {
	// Print an int64 in the half-open interval [0, 100).
	fmt.Println(rand.N(int64(100)))

	// Sleep for a random duration between 0 and 100 milliseconds.
	time.Sleep(rand.N(100 * time.Millisecond))
}

func ExampleShuffle() {
	words := strings.Fields("ink runs from the corners of my mouth")
	rand.Shuffle(len(words), func { i, j -> words[i], words[j] = words[j], words[i] })
	fmt.Println(words)
}

func ExampleShuffle_slicesInUnison() {
	numbers := []byte("12345")
	letters := []byte("ABCDE")
	// Shuffle numbers, swapping corresponding entries in letters at the same time.
	rand.Shuffle(len(numbers), func { i, j ->
		numbers[i], numbers[j] = numbers[j], numbers[i]
		letters[i], letters[j] = letters[j], letters[i]
	})
	for i := range numbers {
		fmt.Printf("%c: %c\n", letters[i], numbers[i])
	}
}

func ExampleIntN() {
	fmt.Println(rand.IntN(100))
	fmt.Println(rand.IntN(100))
	fmt.Println(rand.IntN(100))
}
