// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a version of ../prime.go with type params

// Enough gaps to trigger a map implementation of the method.
// Also includes a duplicate to test that it doesn't cause problems

package main

import "fmt"

// For now, a lone type parameter is not permitted as RHS in a type declaration (issue #45639).
// type Likeint[T interface{ ~int | ~uint8 }] T
type Likeint int

// type Prime2 Likeint[int]
type Prime2 Likeint

const (
	p2  Prime2 = 2
	p3  Prime2 = 3
	p5  Prime2 = 5
	p7  Prime2 = 7
	p77 Prime2 = 7 // Duplicate; note that p77 doesn't appear below.
	p11 Prime2 = 11
	p13 Prime2 = 13
	p17 Prime2 = 17
	p19 Prime2 = 19
	p23 Prime2 = 23
	p29 Prime2 = 29
	p37 Prime2 = 31
	p41 Prime2 = 41
	p43 Prime2 = 43
)

func main() {
	ck(0, "Prime2(0)")
	ck(1, "Prime2(1)")
	ck(p2, "p2")
	ck(p3, "p3")
	ck(4, "Prime2(4)")
	ck(p5, "p5")
	ck(p7, "p7")
	ck(p77, "p7")
	ck(p11, "p11")
	ck(p13, "p13")
	ck(p17, "p17")
	ck(p19, "p19")
	ck(p23, "p23")
	ck(p29, "p29")
	ck(p37, "p37")
	ck(p41, "p41")
	ck(p43, "p43")
	ck(44, "Prime2(44)")
}

func ck(prime Prime2, str string) {
	if fmt.Sprint(prime) != str {
		panic("prime2.go: " + str)
	}
}
