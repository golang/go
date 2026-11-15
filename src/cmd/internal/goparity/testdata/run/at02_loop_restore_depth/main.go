// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
)

func fill(b []byte, seed uint64) {
	for i := range b {
		b[i] = byte(seed*31+uint64(i))%251 + 1
	}
}

func verify(b []byte, seed uint64) bool {
	for i := range b {
		if b[i] != byte(seed*31+uint64(i))%251+1 {
			return false
		}
	}
	return true
}

func main() {
	bad := false
	for o := range 6 {
		outer := make([]byte, 4096+o)
		fill(outer, uint64(o)+1)
	inner:
		for j := range 60 {
			inner := make([]byte, 2048+j%512)
			fill(inner, uint64(j)+100)
			if j%2 == 0 {
				continue inner
			}
			_ = inner[0]
		}
		if !verify(outer, uint64(o)+1) {
			fmt.Printf("LAW outer iteration %d corrupted\n", o)
			bad = true
		}
	}
	if bad {
		os.Exit(1)
	}
	fmt.Println("ok")
}
