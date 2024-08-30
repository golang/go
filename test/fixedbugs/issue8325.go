// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8325: corrupted byte operations during optimization
// pass.

package main

const alphanum = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

func main() {
	var bytes = []byte{10, 20, 30, 40, 50}

	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}

	for _, b := range bytes {
		switch {
		case '0' <= b && b <= '9',
			'A' <= b && b <= 'Z':
		default:
			println("found a bad character", string(b))
			panic("BUG")
		}

	}
}
