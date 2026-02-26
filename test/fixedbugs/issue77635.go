// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77635: test building values of zero-sized types.

package main

import "reflect"

func F[T interface{ [2][0]int }](x T) bool {
    return reflect.DeepEqual(struct {
        t T
        c chan int
    }{t: x}, 1)
}

func main() {
    var t [2][0]int
    F(t)
}
