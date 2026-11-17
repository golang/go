// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"runtime/debug"
)

const depth = 20000

func fill(buf []byte) int64 {
	var s int64
	for i := range buf {
		buf[i] = 1
		s += int64(buf[i])
	}
	return s
}

func selfRec(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	selfRec(n-1, k, acc)
}

func ping(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	pong(n-1, k, acc)
}

func pong(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	ping(n-1, k, acc)
}

func alpha(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	beta(n-1, k, acc)
}

func beta(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	gamma(n-1, k, acc)
}

func gamma(n, k int, acc *int64) {
	if n == 0 {
		return
	}
	buf := make([]byte, k)
	*acc += fill(buf)
	alpha(n-1, k, acc)
}

func main() {
	debug.SetMaxStack(64 << 20)
	n := len(os.Args)
	k := 4096 * n / n
	var acc int64
	selfRec(depth, k, &acc)
	ping(depth, k, &acc)
	alpha(depth, k, &acc)
	want := int64(3*depth) * int64(k)
	if acc != want {
		fmt.Printf("LAW acc=%d want=%d\n", acc, want)
		os.Exit(1)
	}
	fmt.Println("ok")
}
