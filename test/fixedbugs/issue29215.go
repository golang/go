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
