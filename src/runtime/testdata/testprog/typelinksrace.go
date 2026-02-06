// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"reflect"
)


func init() {
	register("TypelinksRace", TypelinksRace)
}

const N = 2

type T int

// just needs some exotic type that the compiler doesn't build its pointer type
var t = reflect.TypeOf([5]T{})

var ch = make(chan int, N)

func TypelinksRace() {
	for range N {
		go func() {
			_ = reflect.PointerTo(t)
			ch <- 1
		}()
	}
	for range N {
		<-ch
	}
	fmt.Println("OK")
}
