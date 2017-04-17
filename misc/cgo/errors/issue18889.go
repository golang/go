package main

import "C"

func main() {
	_ = C.malloc // ERROR HERE
}
