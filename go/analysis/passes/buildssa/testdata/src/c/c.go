// Package c is to test buildssa importing packages.
package c

import (
	"a"
	"b"
	"unsafe"
)

func A() {
	_ = a.Fib(10)
}

func B() {
	var x int
	ptr := unsafe.Pointer(&x)
	_ = b.LoadPointer(&ptr)

	m := b.G.Load()
	f := b.Load(&b.G)
	if f != m {
		panic("loads of b.G are expected to be indentical")
	}
}
