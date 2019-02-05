package main

import (
	"fmt"
	"os"
)

func test() {
	pwd, err := os.Getwd()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	fmt.Println(pwd)
}

func main() {
	growstack() // Use stack early to prevent growth during test, which confuses gdb
	test()
}

var snk string

//go:noinline
func growstack() {
	snk = fmt.Sprintf("%#v,%#v,%#v", 1, true, "cat")
}
