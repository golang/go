// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check all ways to compare a non-constant string to the empty string.

package main

import (
	"fmt"
	"os"
)

var (
	s      = "abc"
	e      = ""
	failed bool
)

func main() {
	want(true, "" < s, `"" < s`)
	want(false, s < "", `s < ""`)
	want(false, "" < e, `"" < e`)
	want(false, e < "", `e < ""`)

	want(true, "" <= s, `"" <= s`)
	want(false, s <= "", `s <= ""`)
	want(true, "" <= e, `"" <= e`)
	want(true, e <= "", `e <= ""`)

	want(false, "" > s, `"" > s`)
	want(true, s > "", `s > ""`)
	want(false, "" > e, `"" > e`)
	want(false, e > "", `e > ""`)

	want(false, "" >= s, `"" >= s`)
	want(true, s >= "", `s >= ""`)
	want(true, "" >= e, `"" >= e`)
	want(true, e >= "", `e >= ""`)

	want(false, "" == s, `"" == s`)
	want(false, s == "", `s == ""`)
	want(true, "" == e, `"" == e`)
	want(true, e == "", `e == ""`)

	want(true, "" != s, `"" != s`)
	want(true, s != "", `s != ""`)
	want(false, "" != e, `"" != e`)
	want(false, e != "", `e != ""`)

	if failed {
		os.Exit(1)
	}
}

//go:noinline
func want(b bool, have bool, msg string) {
	if b != have {
		fmt.Println(msg)
		failed = true
	}
}
