// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 5515: miscompilation doing inlining in generated method wrapper

package main

type T uint32

func main() {
        b := make([]T, 8)
        b[0] = 0xdeadbeef
        rs := Slice(b)
        sort(rs)
}

type Slice []T

func (s Slice) Swap(i, j int) {
        tmp := s[i]
        s[i] = s[j]
        s[j] = tmp
}

type Interface interface {
        Swap(i, j int)
}

func sort(data Interface) {
        data.Swap(0, 4)
}
