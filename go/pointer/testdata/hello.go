//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
)

type S int

var theS S

func (s *S) String() string {
	print(s) // @pointsto command-line-arguments.theS
	return ""
}

func main() {
	// os.Args is considered intrinsically allocated,
	// but may also be set explicitly (e.g. on Windows), hence '...'.
	print(os.Args) // @pointsto <command-line args> | ...
	fmt.Println("Hello, World!", &theS)
}

// @calls command-line-arguments.main               -> fmt.Println
// @calls (*fmt.pp).handleMethods -> (*command-line-arguments.S).String
