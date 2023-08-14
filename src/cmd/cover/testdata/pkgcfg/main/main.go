package main

import (
	"cfg/a"
	"cfg/b"
)

func main() {
	a.A(2)
	a.A(1)
	a.A(0)
	b.B(1)
	b.B(0)
	println("done")
}
