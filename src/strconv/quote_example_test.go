// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"fmt"
	"strconv"
)

func ExampleUnquote() {
	test := func(s string) {
		t, err := strconv.Unquote(s)
		if err != nil {
			fmt.Printf("Unquote(%#v): %v\n", s, err)
		} else {
			fmt.Printf("Unquote(%#v) = %v\n", s, t)
		}
	}

	s := `cafe\u0301`
	// If the string doesn't have quotes, it can't be unquoted.
	test(s) // invalid syntax
	test("`" + s + "`")
	test(`"` + s + `"`)

	test(`'\u00e9'`)

	// Output:
	// Unquote("cafe\\u0301"): invalid syntax
	// Unquote("`cafe\\u0301`") = cafe\u0301
	// Unquote("\"cafe\\u0301\"") = café
	// Unquote("'\\u00e9'") = é
}
