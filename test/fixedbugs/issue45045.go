// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"runtime"
	"unsafe"
)

func k(c chan string, val string) string {
	b := make([]byte, 1000)
	runtime.SetFinalizer(&b[0], func(*byte) {
		c <- val
	})
	var s string
	h := (*reflect.StringHeader)(unsafe.Pointer(&s))
	h.Data = uintptr(unsafe.Pointer(&b[0]))
	h.Len = len(b)
	return s
}

func main() {
	{
		c := make(chan string, 2)
		m := make(map[string]int)
		m[k(c, "first")] = 0
		m[k(c, "second")] = 0
		runtime.GC()
		if s := <-c; s != "first" {
			panic("map[string], second key did not retain.")
		}
		runtime.KeepAlive(m)
	}

	{
		c := make(chan string, 2)
		m := make(map[[2]string]int)
		m[[2]string{k(c, "first")}] = 0
		m[[2]string{k(c, "second")}] = 0
		runtime.GC()
		if s := <-c; s != "first" {
			panic("map[[2]string], second key did not retain.")
		}
		runtime.KeepAlive(m)
	}
}
