package main

import "reflect"

func main() {
	// Regression test for issue 9462.
	got := reflect.SliceOf(reflect.TypeOf(byte(0))).String()
	if got != "[]uint8" && got != "[]byte" { // result varies by toolchain
		println("BUG: " + got)
	}
}
