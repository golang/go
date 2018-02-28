package main

import (
	_ "complex/nest/sub/test12"
	_ "complex/nest/sub/test23"
	"complex/w"
	"v"
)

func main() {
	println(v.Hello + " " + w.World)
}
