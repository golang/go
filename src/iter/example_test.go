// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iter_test

import (
	"fmt"
	"iter"
	"slices"
	"strings"
)

func ExampleSeq() {
	// A Seq is a function that passes each value of a sequence to
	// yield. Nothing requires the sequence to be finite: this one ends
	// only when yield reports that the consumer has stopped, which
	// happens when the range loop below breaks.
	var fib iter.Seq[int] = func(yield func(int) bool) {
		a, b := 0, 1
		for yield(a) {
			a, b = b, a+b
		}
	}

	for n := range fib {
		if n > 50 {
			break
		}
		fmt.Print(n, " ")
	}
	fmt.Println()

	// Output:
	// 0 1 1 2 3 5 8 13 21 34
}

func ExampleSeq2() {
	// A Seq2 passes a pair of values per element, here the two halves
	// of a key=value field. Ranging over an existing Seq, in this case
	// strings.SplitSeq, keeps the fields from being collected into a
	// slice first.
	var attrs iter.Seq2[string, string] = func(yield func(string, string) bool) {
		for field := range strings.SplitSeq("gopher=blue,size=large,tail", ",") {
			k, v, _ := strings.Cut(field, "=")
			if !yield(k, v) {
				return
			}
		}
	}

	for k, v := range attrs {
		fmt.Printf("%q %q\n", k, v)
	}

	// Output:
	// "gopher" "blue"
	// "size" "large"
	// "tail" ""
}

func ExamplePull() {
	// Merging two sorted sequences means advancing whichever one holds
	// the smaller value, so the two have to move independently. A range
	// loop cannot do that, since it drives a single sequence from start
	// to finish. Pull converts each sequence into a next function that
	// produces one value per call.
	next1, stop1 := iter.Pull(slices.Values([]int{1, 3, 5, 7}))
	defer stop1()
	next2, stop2 := iter.Pull(slices.Values([]int{2, 3, 6}))
	defer stop2()

	v1, ok1 := next1()
	v2, ok2 := next2()
	for ok1 || ok2 {
		if !ok2 || ok1 && v1 <= v2 {
			fmt.Print(v1, " ")
			v1, ok1 = next1()
		} else {
			fmt.Print(v2, " ")
			v2, ok2 = next2()
		}
	}
	fmt.Println()

	// Output:
	// 1 2 3 3 5 6 7
}

func ExamplePull2() {
	// This caller reads only the first two pairs and leaves the rest of
	// the sequence unread. Because the sequence is not consumed to
	// completion, stop has to be called to let the iterator function
	// finish and return.
	next, stop := iter.Pull2(slices.All([]string{"hydrogen", "helium", "lithium"}))
	defer stop()

	for range 2 {
		i, element, ok := next()
		if !ok {
			break
		}
		fmt.Println(i, element)
	}

	// Output:
	// 0 hydrogen
	// 1 helium
}
