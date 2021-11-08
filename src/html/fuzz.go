// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gofuzz

package html

import (
	"fmt"
)

func Fuzz(data []byte) int {
	v := string(data)

	e := EscapeString(v)
	u := UnescapeString(e)
	if v != u {
		fmt.Printf("v = %q\n", v)
		fmt.Printf("e = %q\n", e)
		fmt.Printf("u = %q\n", u)
		panic("not equal")
	}

	// As per the documentation, this isn't always equal to v, so it makes
	// no sense to check for equality. It can still be interesting to find
	// panics in it though.
	EscapeString(UnescapeString(v))

	return 0
}
