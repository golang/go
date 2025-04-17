package main

import (
	"./d"
	"fmt"
)

func main() {
	if got, want := d.BuildInt(), 42; got != want {
		panic(fmt.Sprintf("got %d, want %d", got, want))
	}
}
