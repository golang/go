package main

/*
#cgo LDFLAGS: -c

void test() {
	xxx;		// This is line 7.
}
*/
import "C"

func main() {
	C.test()
}
