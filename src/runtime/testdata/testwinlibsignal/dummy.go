// +build windows

package main

//export Dummy
func Dummy() int {
	return 42
}

func main() {}
