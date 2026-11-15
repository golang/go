// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime"
	"weak"
)

func checkBytes(n int) int {
	s := make([]byte, n)
	for i := range s {
		s[i] = byte(i*31 + 7)
	}
	runtime.SetFinalizer(&s[0], func(*byte) {})
	var cleaned bool
	runtime.AddCleanup(&s[0], func(*bool) {}, &cleaned)
	w := weak.Make(&s[0])
	if w.Value() != &s[0] {
		fmt.Println("LAW weak value mismatch", n)
		return 1
	}
	var sum int64
	for _, b := range s {
		sum += int64(b)
	}
	fmt.Println("finalizer_make bytes ok", n, sum)
	runtime.KeepAlive(s)
	runtime.KeepAlive(cleaned)
	return 0
}

func checkInts(n int) int {
	s := make([]int64, n)
	for i := range s {
		s[i] = int64(i*1000 + 13)
	}
	runtime.SetFinalizer(&s[0], func(*int64) {})
	var cleaned bool
	runtime.AddCleanup(&s[0], func(*bool) {}, &cleaned)
	w := weak.Make(&s[0])
	if w.Value() != &s[0] {
		fmt.Println("LAW weak value mismatch", n)
		return 1
	}
	var sum int64
	for _, v := range s {
		sum += v
	}
	fmt.Println("finalizer_make ints ok", n, sum)
	runtime.KeepAlive(s)
	runtime.KeepAlive(cleaned)
	return 0
}

func main() {
	code := 0
	for _, n := range []int{40, 300, 5000, 60000} {
		if checkBytes(n) != 0 {
			code = 1
		}
	}
	for _, n := range []int{8, 64, 800, 7000} {
		if checkInts(n) != 0 {
			code = 1
		}
	}
	os.Exit(code)
}
