// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter_test

import (
	"fmt"
	"os"
	"text/tabwriter"
)

func ExampleWriter_Init() {
	w := new(tabwriter.Writer)

	// Format in tab-separated columns with a tab stop of 8.
	w.Init(os.Stdout, 0, 8, 0, '\t', 0)
	fmt.Fprintln(w, "a\tb\tc\td\t.")
	fmt.Fprintln(w, "123\t12345\t1234567\t123456789\t.")
	fmt.Fprintln(w)
	w.Flush()

	// Format right-aligned in space-separated columns of minimal width 5
	// and at least one blank of padding (so wider column entries do not
	// touch each other).
	w.Init(os.Stdout, 5, 0, 1, ' ', tabwriter.AlignRight)
	fmt.Fprintln(w, "a\tb\tc\td\t.")
	fmt.Fprintln(w, "123\t12345\t1234567\t123456789\t.")
	fmt.Fprintln(w)
	w.Flush()

	// output:
	// a	b	c	d		.
	// 123	12345	1234567	123456789	.
	//
	//     a     b       c         d.
	//   123 12345 1234567 123456789.
}

func Example_elastic() {
	// Observe how the b's and the d's, despite appearing in the
	// second cell of each line, belong to different columns.
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 1, '.', tabwriter.AlignRight|tabwriter.Debug)
	fmt.Fprintln(w, "a\tb\tc")
	fmt.Fprintln(w, "aa\tbb\tcc")
	fmt.Fprintln(w, "aaa\t") // trailing tab
	fmt.Fprintln(w, "aaaa\tdddd\teeee")
	w.Flush()

	// output:
	// ....a|..b|c
	// ...aa|.bb|cc
	// ..aaa|
	// .aaaa|.dddd|eeee
}

func Example_trailingTab() {
	// Observe that the third line has no trailing tab,
	// so its final cell is not part of an aligned column.
	const padding = 3
	w := tabwriter.NewWriter(os.Stdout, 0, 0, padding, '-', tabwriter.AlignRight|tabwriter.Debug)
	fmt.Fprintln(w, "a\tb\taligned\t")
	fmt.Fprintln(w, "aa\tbb\taligned\t")
	fmt.Fprintln(w, "aaa\tbbb\tunaligned") // no trailing tab
	fmt.Fprintln(w, "aaaa\tbbbb\taligned\t")
	w.Flush()

	// output:
	// ------a|------b|---aligned|
	// -----aa|-----bb|---aligned|
	// ----aaa|----bbb|unaligned
	// ---aaaa|---bbbb|---aligned|
}
