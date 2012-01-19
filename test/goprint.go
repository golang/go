// $G $D/$F.go && $L $F.$A && ./$A.out 2>&1 | cmp - $D/$F.out

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "time"

func main() {
	go println(42, true, false, true, 1.5, "world", (chan int)(nil), []int(nil), (map[string]int)(nil), (func())(nil), byte(255))
	time.Sleep(1e6)
}
