// run

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure the text output for bounds check failures is as expected.

package main

import (
	"fmt"
	"os"
	"runtime"
	"text/tabwriter"
)

// Testing with length 3 slices, arrays, and strings.
// Large (>1<<32) values are included to test 32-bit platforms.
var indexes = []int64{-9876543210, -1, 0, 2, 3, 9876543210}
var slices = []int64{-9876543210, -1, 0, 3, 4, 9876543210}

var w *tabwriter.Writer

func main() {
	w = tabwriter.NewWriter(os.Stdout, 0, 0, 1, ' ', tabwriter.AlignRight)
	defer w.Flush()
	doIndex()
	doSlice()
	doSlice3()
}
func doIndex() {
	a := []int{1, 2, 3}
	for _, i := range indexes {
		printPanic(fmt.Sprintf("slice[%d]", i), func() {
			_ = a[i]
		})
	}
	b := [3]int{1, 2, 3}
	for _, i := range indexes {
		printPanic(fmt.Sprintf("array[%d]", i), func() {
			_ = b[i]
		})
	}
	c := "123"
	for _, i := range indexes {
		printPanic(fmt.Sprintf("string[%d]", i), func() {
			_ = c[i]
		})
	}
}

func doSlice() {
	a := []int{1, 2, 3}
	for _, i := range slices {
		for _, j := range slices {
			printPanic(fmt.Sprintf("slice[%d:%d]", i, j), func() {
				_ = a[i:j]
			})
		}
	}
	b := [3]int{1, 2, 3}
	for _, i := range slices {
		for _, j := range slices {
			printPanic(fmt.Sprintf("array[%d:%d]", i, j), func() {
				_ = b[i:j]
			})
		}
	}
	c := "123"
	for _, i := range slices {
		for _, j := range slices {
			printPanic(fmt.Sprintf("string[%d:%d]", i, j), func() {
				_ = c[i:j]
			})
		}
	}
}

func doSlice3() {
	a := []int{1, 2, 3}
	for _, i := range slices {
		for _, j := range slices {
			for _, k := range slices {
				printPanic(fmt.Sprintf("slice[%d:%d:%d]", i, j, k), func() {
					_ = a[i:j:k]
				})
			}
		}
	}
	b := [3]int{1, 2, 3}
	for _, i := range slices {
		for _, j := range slices {
			for _, k := range slices {
				printPanic(fmt.Sprintf("array[%d:%d:%d]", i, j, k), func() {
					_ = b[i:j:k]
				})
			}
		}
	}
}

func printPanic(msg string, f func()) {
	defer func() {
		res := "no panic"
		if e := recover(); e != nil {
			res = e.(runtime.Error).Error()
		}
		fmt.Fprintf(w, "%s\t %s\n", msg, res)
	}()
	f()
}
