// compile

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f() {
        var s string
        var p, q bool
        s = "a"
        for p {
                p = false == (true != q)
                s = ""
        }
        _ = s == "bbb"
}

// Another case: load from negative offset of a symbol
// in dead code (issue 30257).
func g() {
	var i int
	var s string

	if true {
		s = "a"
	}

	if f := 0.0; -f < 0 {
		i = len(s[:4])
	}

	_ = s[i-1:0] != "bb" && true
}
