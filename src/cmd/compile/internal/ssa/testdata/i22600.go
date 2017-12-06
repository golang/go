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
	test()
}
