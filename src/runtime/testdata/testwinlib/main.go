//go:build windows && cgo
// +build windows,cgo

package main

// #include <windows.h>
// typedef void(*callmeBackFunc)();
// static void bridgeCallback(callmeBackFunc callback) {
//	callback();
//}
import "C"

// CallMeBack call backs C code.
//export CallMeBack
func CallMeBack(callback C.callmeBackFunc) {
	C.bridgeCallback(callback)
}

// Dummy is called by the C code before registering the exception/continue handlers simulating a debugger.
// This makes sure that the Go runtime's lastcontinuehandler is reached before the C continue handler and thus,
// validate that it does not crash the program before another handler could take an action.
// The idea here is to reproduce what happens when you attach a debugger to a running program.
// It also simulate the behavior of the .Net debugger, which register its exception/continue handlers lazily.
//export Dummy
func Dummy() int {
	return 42
}

func main() {}
