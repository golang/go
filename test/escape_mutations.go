// errorcheck -0 -m -d=escapemutationscalls,zerocopy -l

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import "fmt"

type B struct {
	x  int
	px *int
	pb *B
}

func F1(b *B) { // ERROR "mutates param: b derefs=0"
	b.x = 1
}

func F2(b *B) { // ERROR "mutates param: b derefs=1"
	*b.px = 1
}

func F2a(b *B) { // ERROR "mutates param: b derefs=0"
	b.px = nil
}

func F3(b *B) { // ERROR "leaking param: b"
	fmt.Println(b) // ERROR "\.\.\. argument does not escape"
}

func F4(b *B) { // ERROR "leaking param content: b"
	fmt.Println(*b) // ERROR "\.\.\. argument does not escape" "\*b escapes to heap"
}

func F4a(b *B) { // ERROR "leaking param content: b" "mutates param: b derefs=0"
	b.x = 2
	fmt.Println(*b) // ERROR "\.\.\. argument does not escape" "\*b escapes to heap"
}

func F5(b *B) { // ERROR "leaking param: b"
	sink = b
}

func F6(b *B) int { // ERROR "b does not escape, mutate, or call"
	return b.x
}

var sink any

func M() {
	var b B // ERROR "moved to heap: b"
	F1(&b)
	F2(&b)
	F2a(&b)
	F3(&b)
	F4(&b)
}

func g(s string) { // ERROR "s does not escape, mutate, or call"
	sink = &([]byte(s))[10] // ERROR "\(\[\]byte\)\(s\) escapes to heap"
}

func h(out []byte, s string) { // ERROR "mutates param: out derefs=0" "s does not escape, mutate, or call"
	copy(out, []byte(s)) // ERROR "zero-copy string->\[\]byte conversion" "\(\[\]byte\)\(s\) does not escape"
}

func i(s string) byte { // ERROR "s does not escape, mutate, or call"
	p := []byte(s) // ERROR "zero-copy string->\[\]byte conversion" "\(\[\]byte\)\(s\) does not escape"
	return p[20]
}

func j(s string, x byte) { // ERROR "s does not escape, mutate, or call"
	p := []byte(s) // ERROR "\(\[\]byte\)\(s\) does not escape"
	p[20] = x
}
