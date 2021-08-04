package main

import "C"

// This program will crash.
// We want to test unwinding from a cgo callback.

/*
void panic_callback();

static void call_callback(void) {
	panic_callback();
}
*/
import "C"

func init() {
	register("PanicCallback", PanicCallback)
}

//export panic_callback
func panic_callback() {
	var i *int
	*i = 42
}

func PanicCallback() {
	C.call_callback()
}
