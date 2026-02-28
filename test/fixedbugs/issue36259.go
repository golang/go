// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func rotate(s []int, m int) {
    l := len(s)
    m = m % l
    buf := make([]int, m)

    copy(buf, s)
    copy(s, s[m:])
    copy(s[l-m:], buf)
}

func main() {
    a0 := [...]int{1,2,3,4,5}
    println(a0[0])

    rotate(a0[:], 1)
    println(a0[0])

    rotate(a0[:], -3)
    println(a0[0])
}
