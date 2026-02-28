// errorcheck

// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The compiler was failing to correctly report an error when a dot
// expression was used a struct literal key.

package p

type T struct {
        toInt    map[string]int
        toString map[int]string
}

var t = T{
        foo.toInt:    make(map[string]int), // ERROR "field name"
        bar.toString: make(map[int]string), // ERROR "field name"
}
