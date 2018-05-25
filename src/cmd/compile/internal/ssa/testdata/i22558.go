package main

import (
	"fmt"
	"os"
)

type big struct {
	pile [768]int8
}

type thing struct {
	name  string
	next  *thing
	self  *thing
	stuff []big
}

func test(t *thing, u *thing) {
	if t.next != nil {
		return
	}
	fmt.Fprintf(os.Stderr, "%s\n", t.name)
	u.self = u
	t.self = t
	t.next = u
	for _, p := range t.stuff {
		if isFoo(t, p) {
			return
		}
	}
}

//go:noinline
func isFoo(t *thing, b big) bool {
	return true
}

func main() {
	growstack() // Use stack early to prevent growth during test, which confuses gdb
	t := &thing{name: "t", self: nil, next: nil, stuff: make([]big, 1)}
	u := thing{name: "u", self: t, next: t, stuff: make([]big, 1)}
	test(t, &u)
}

var snk string

//go:noinline
func growstack() {
	snk = fmt.Sprintf("%#v,%#v,%#v", 1, true, "cat")
}
