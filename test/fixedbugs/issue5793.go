// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 5793: calling 2-arg builtin with multiple-result f() call expression gives
// spurious error.

package main

func complexArgs() (float64, float64) {
	return 5, 7
}

func appendArgs() ([]string, string) {
	return []string{"foo"}, "bar"
}

func appendMultiArgs() ([]byte, byte, byte) {
	return []byte{'a', 'b'}, '1', '2'
}

func main() {
	if c := complex(complexArgs()); c != 5+7i {
		panic(c)
	}

	if s := append(appendArgs()); len(s) != 2 || s[0] != "foo" || s[1] != "bar" {
		panic(s)
	}

	if b := append(appendMultiArgs()); len(b) != 4 || b[0] != 'a' || b[1] != 'b' || b[2] != '1' || b[3] != '2' {
		panic(b)
	}
}
