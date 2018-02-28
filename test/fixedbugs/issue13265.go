// errorcheck -0 -race

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 13265: nil pointer deref.

package p

func f() {
    var c chan chan chan int
    for ; ; <-<-<-c {
    }
}
