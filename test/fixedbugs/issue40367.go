// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func case1() {
	rates := []int32{1,2,3,4,5,6}
	var sink [6]int
	j := len(sink)
	for star, _ := range rates {
		if star+1 < 1 {
			panic("")
		}
		j--
		sink[j] = j
	}
}

func case2() {
	i := 0
	var sink [3]int
	j := len(sink)
top:
	j--
	sink[j] = j
	if i < 2 {
		i++
		if i < 1 {
			return
		}
		goto top
	}
}

func main() {
	case1()
	case2()
}