// +build ignore

package main

import "unsafe"

var a int

func conv1() {
	// Conversions of channel direction.
	ch := make(chan int)    // @line c1make
	print((<-chan int)(ch)) // @pointsto makechan@c1make:12
	print((chan<- int)(ch)) // @pointsto makechan@c1make:12
}

func conv2() {
	// []byte/[]rune literal
	print([]byte("foo")) // @pointsto "foo":[]byte
	print([]rune("bar")) // @pointsto "bar":[]rune

	// string -> []byte/[]rune conversion
	s := "foo"
	ba := []byte(s) // @line c2ba
	ra := []rune(s) // @line c2ra
	print(ba)       // @pointsto convert@c2ba:14
	print(ra)       // @pointsto convert@c2ra:14
}

func conv3() {
	// Conversion of same underlying types.
	type PI *int
	pi := PI(&a)
	print(pi) // @pointsto main.a

	pint := (*int)(pi)
	print(pint) // @pointsto main.a

	// Conversions between pointers to identical base types.
	var y *PI = &pi
	var x **int = (**int)(y)
	print(*x) // @pointsto main.a
	print(*y) // @pointsto main.a
	y = (*PI)(x)
	print(*y) // @pointsto main.a
}

// @warning "main.conv4 contains an unsafe.Pointer conversion"
func conv4() {
	// Handling of unsafe.Pointer conversion is unsound:
	// we lose the alias to main.a and get something like new(int) instead.
	// We require users to provide aliasing summaries.
	p := (*int)(unsafe.Pointer(&a)) // @line c2p
	print(p)                        // @pointsto convert@c2p:13
}

func main() {
	conv1()
	conv2()
	conv3()
	conv4()
}
