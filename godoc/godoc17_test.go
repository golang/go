// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.7
// +build go1.7

package godoc

import (
	"bytes"
	"fmt"
	"testing"
)

// Verify that scanIdentifier isn't quadratic.
// This doesn't actually measure and fail on its own, but it was previously
// very obvious when running by hand.
//
// TODO: if there's a reliable and non-flaky way to test this, do so.
// Maybe count user CPU time instead of wall time? But that's not easy
// to do portably in Go.
func TestStructField(t *testing.T) {
	for _, n := range []int{10, 100, 1000, 10000} {
		n := n
		t.Run(fmt.Sprint(n), func(t *testing.T) {
			var buf bytes.Buffer
			fmt.Fprintf(&buf, "package foo\n\ntype T struct {\n")
			for i := 0; i < n; i++ {
				fmt.Fprintf(&buf, "\t// Field%d is foo.\n\tField%d int\n\n", i, i)
			}
			fmt.Fprintf(&buf, "}\n")
			linkifySource(t, buf.Bytes())
		})
	}
}
