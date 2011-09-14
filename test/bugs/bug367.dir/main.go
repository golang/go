package main

import (
	"./p"
)

type T struct{ *p.S }

func main() {
	var t T
	p.F(t)
}
