// run

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func shouldPanic(f func()) {
        defer func() {
                if recover() == nil {
                        panic("not panicking")
                }
        }()
        f()
}

func f() {
        length := int(^uint(0) >> 1)
        a := make([]struct{}, length)
        b := make([]struct{}, length)
        _ = append(a, b...)
}

func main() {
	shouldPanic(f)
}
