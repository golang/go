// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test switch statements that get compiled to jump tables.

package main

import "runtime"

var failed bool

func check(name string, x, got, want int64) {
	if got != want {
		println(name, x, "got", got, "want", want)
		failed = true
	}
}

//go:noinline
func dense(x, y int64) int64 {
	switch x {
	case 0:
		return y + 3
	case 1:
		return y - 14
	case 2:
		return y * 15
	case 3:
		return y ^ 92
	case 4:
		return y | 65
	case 5:
		return y &^ 35
	case 6:
		return y << 2
	case 7:
		return -y
	case 8:
		return y >> 1
	case 9:
		return y * y
	}
	return -1
}

func testDense() {
	y := int64(1000)
	want := []int64{
		y + 3, y - 14, y * 15, y ^ 92, y | 65,
		y &^ 35, y << 2, -y, y >> 1, y * y,
	}
	for x := int64(-4); x < 14; x++ {
		w := int64(-1)
		if 0 <= x && x < 10 {
			w = want[x]
		}
		check("dense", x, dense(x, y), w)
	}
	// The upper bits of the value must not get lost.
	for x := int64(0); x < 10; x++ {
		check("dense", x+1<<32, dense(x+1<<32, y), -1)
		check("dense", x-1<<32, dense(x-1<<32, y), -1)
	}
}

//go:noinline
func sparse(x int32, y int64) int64 {
	switch x {
	case -9:
		return y + 1
	case -6, -5:
		return y + 2
	case -2:
		y += 3
		fallthrough
	case 0:
		return y + 4
	default:
		return y - 1
	case 3:
		return y + 5
	case 4:
		return y + 6
	case 7:
		return y + 7
	case 10:
		return y + 8
	case 12:
		return y + 9
	}
}

func testSparse() {
	y := int64(1000)
	want := map[int32]int64{
		-9: y + 1, -6: y + 2, -5: y + 2, -2: y + 7, 0: y + 4,
		3: y + 5, 4: y + 6, 7: y + 7, 10: y + 8, 12: y + 9,
	}
	for x := int32(-14); x < 18; x++ {
		w, ok := want[x]
		if !ok {
			w = y - 1
		}
		check("sparse", int64(x), sparse(x, y), w)
	}
	for _, x := range []int32{-1 << 31, -1<<31 + 9, 1<<31 - 13, 1<<31 - 1} {
		check("sparse", int64(x), sparse(x, y), y-1)
	}
}

// The functions below record the cases that they run as hexadecimal digits.

//go:noinline
func loop(s []byte) int64 {
	var n int64
	for _, c := range s {
		switch c {
		case 0, 2, 4:
			continue
		case 1:
			n = n<<4 | 1
		case 3:
			runtime.Gosched()
			n = n<<4 | 3
		case 5:
			n = n<<4 | 5
		case 6:
			n = n<<4 | 6
		case 7:
			n = n<<4 | 7
		case 8:
			break
		case 9:
			n = n<<4 | 9
		}
		n = n<<4 | 0xf
	}
	return n
}

//go:noinline
func labels(rows [][]int) int64 {
	var n int64
outer:
	for _, row := range rows {
		for _, v := range row {
			switch v {
			case 0:
				continue outer
			case 1:
				continue
			case 2:
				break outer
			case 3:
				n = n<<4 | 3
			case 4:
				n = n<<4 | 4
			case 5:
				n = n<<4 | 5
			case 6:
				n = n<<4 | 6
				continue outer
			case 7:
				n = n<<4 | 7
			case 8:
				n = n<<4 | 8
			}
			n = n<<4 | 0xe
		}
		n = n<<4 | 0xf
	}
	return n
}

//go:noinline
func gotos(prog []int) int64 {
	var n int64
	i := 0
again:
	n = n<<4 | 0xa
other:
	if i == len(prog) {
		return n
	}
	op := prog[i]
	i++
	switch op {
	case 0:
		goto again
	case 1:
		goto other
	case 2:
		n = n<<4 | 2
		goto again
	case 3:
		n = n<<4 | 3
		goto other
	case 4:
		n = n<<4 | 4
	case 5:
		n = n<<4 | 5
	case 6:
		n = n<<4 | 6
	case 7:
		return -n
	}
	n = n<<4 | 0xd
	goto other
}

func testLoops() {
	check("loop", 0, loop([]byte{1, 0, 3, 2, 8, 5, 11, 9, 4, 6, 7}), 0x1f3ff5ff9f6f7f)
	check("loop", 1, loop(nil), 0)
	rows := [][]int{{3, 1, 4}, {5, 0, 7}, {}, {6, 8}, {9, 7}, {8, 2, 3}, {4}}
	check("labels", 0, labels(rows), 0x3e4ef5ef6e7ef8e)
	check("gotos", 0, gotos([]int{0, 1, 2, 3, 4, 9, 5, 1, 6}), 0xaa2a34dd5d6d)
	check("gotos", 1, gotos([]int{4, 7}), -0xa4d)
}

//go:noinline
func lengths(s string, y int64) int64 {
	switch s {
	case "a":
		return y + 1
	case "bb":
		return y + 2
	case "ccc":
		return y + 3
	case "dddd":
		return y + 4
	case "eeeee":
		return y + 5
	case "ffffff":
		return y + 6
	case "ggggggg":
		return y + 7
	case "hhhhhhhh":
		return y + 8
	}
	return y
}

func testLengths() {
	y := int64(1000)
	strs := []string{
		"", "a", "bb", "ccc", "dddd",
		"eeeee", "ffffff", "ggggggg", "hhhhhhhh", "iiiiiiiii",
	}
	for i, s := range strs {
		w := y
		if 1 <= i && i <= 8 {
			w = y + int64(i)
		}
		check("lengths", int64(i), lengths(s, y), w)
		check("lengths", int64(i), lengths(s+"x", y), y)
	}
}

func main() {
	testDense()
	testSparse()
	testLoops()
	testLengths()
	if failed {
		panic("fail")
	}
}
