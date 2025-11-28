// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"testing"
)

func TestPrintQuoted(t *testing.T) {
	for _, tbl := range []struct {
		in, expected string
	}{
		{in: "baz", expected: `"baz"`},
		{in: "foobar", expected: `"foobar"`},
		// make sure newlines get escaped
		{in: "baz\n", expected: `"baz\n"`},
		// make sure null and escape bytes are properly escaped
		{in: "b\033it", expected: `"b\x1bit"`},
		{in: "b\000ar", expected: `"b\x00ar"`},
		// verify that simple 16-bit unicode runes are escaped with \u, including a greek upper-case sigma and an arbitrary unicode character.
		{in: "\u1234Σ", expected: `"\u1234\u03a3"`},
		// verify that 32-bit unicode runes are escaped with \U along with tabs
		{in: "fizz\tle", expected: `"fizz\tle"`},
		{in: "\U00045678boop", expected: `"\U00045678boop"`},
		// verify carriage returns and backslashes get escaped along with our nulls, newlines and a 32-bit unicode character
		{in: "fiz\\zl\re", expected: `"fiz\\zl\re"`},
	} {
		t.Run(tbl.in, func(t *testing.T) {
			out := runtime.DumpPrintQuoted(tbl.in)
			if out != tbl.expected {
				t.Errorf("unexpected output for print(escaped(%q));\n got: %s\nwant: %s", tbl.in, out, tbl.expected)
			}
		})
	}
}
