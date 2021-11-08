//go:build windows
// +build windows

package main

import "C"

//export Dummy
func Dummy() int {
	return 42
}

func main() {}
