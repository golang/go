package main

// This program will crash.
// We want to test unwinding from a cgo callback.

/*
void call_callback(void);
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
