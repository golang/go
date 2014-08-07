package main

// Tests of range loops.

import "fmt"

// Range over string.
func init() {
	if x := len("Hello, 世界"); x != 13 { // bytes
		panic(x)
	}
	var indices []int
	var runes []rune
	for i, r := range "Hello, 世界" {
		runes = append(runes, r)
		indices = append(indices, i)
	}
	if x := fmt.Sprint(runes); x != "[72 101 108 108 111 44 32 19990 30028]" {
		panic(x)
	}
	if x := fmt.Sprint(indices); x != "[0 1 2 3 4 5 6 7 10]" {
		panic(x)
	}
	s := ""
	for _, r := range runes {
		s = fmt.Sprintf("%s%c", s, r)
	}
	if s != "Hello, 世界" {
		panic(s)
	}

	var x int
	for range "Hello, 世界" {
		x++
	}
	if x != len(indices) {
		panic(x)
	}
}

// Regression test for range of pointer to named array type.
func init() {
	type intarr [3]int
	ia := intarr{1, 2, 3}
	var count int
	for _, x := range &ia {
		count += x
	}
	if count != 6 {
		panic(count)
	}
}

func main() {
}
