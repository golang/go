// +build windows,cgo

package main

// #include <windows.h>
// typedef void(*callmeBackFunc)();
// static void bridgeCallback(callmeBackFunc callback) {
//	callback();
//}
import "C"

// CallMeBack call backs C code
//export CallMeBack
func CallMeBack(callback C.callmeBackFunc) {
	C.bridgeCallback(callback)
}

// Dummy is there to bootstrap the go runtime before setting up the "debugger" continue handlers
//export Dummy
func Dummy() int {
	return 42
}

func main() {}
