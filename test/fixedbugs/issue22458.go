// compile

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Make sure KeepAlive introduces a use of the spilled variable.

package main

import "runtime"

type node struct {
        next *node
}

var x bool

func main() {
        var head *node
        for x {
                head = &node{head}
        }

        runtime.KeepAlive(head)
}
