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
	// Typically a non-fixed seed should be used, such as time.Now().UnixNano().
	// Using a fixed seed will produce the same output on every run.
	r := rand.New(rand.NewSource(99))

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
	// Float32     0.73793465          0.38461488          0.9940225
	// Float64     0.6919607852308565  0.29140004584133117 0.2262092163027547
	// ExpFloat64  0.10400903165715357 0.28855743344575835 0.20489656480442942
	// NormFloat64 -0.5602299711828513 -0.9211692958208376 -1.4262061075859056
	// Int32       1817075958          91420417            1486590581
	// Int64       5724354148158589552 5239846799706671610 5927547564735367388
	// Uint32      2295813601          961197529           3493134579
	// IntN(10)    4                   5                   1
	// Int32N(10)  8                   5                   4
	// Int64N(10)  2                   6                   3
	// Perm        [3 4 2 1 0]         [4 1 2 0 3]         [0 2 1 3 4]
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
	rand.Shuffle(len(words), func(i, j int) {
		words[i], words[j] = words[j], words[i]
	})
	fmt.Println(words)
}

func ExampleShuffle_slicesInUnison() {
	numbers := []byte("12345")
	letters := []byte("ABCDE")
	// Shuffle numbers, swapping corresponding entries in letters at the same time.
	rand.Shuffle(len(numbers), func(i, j int) {
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
