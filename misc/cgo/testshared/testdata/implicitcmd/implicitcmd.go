package main

import (
	"testshared/explicit"
	"testshared/implicit"
)

func main() {
	println(implicit.I() + explicit.E())
}
