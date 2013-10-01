// +build ignore

package main

import (
	"fmt"
	"os"
)

type S int

var theS S

func (s *S) String() string {
	print(s) // @pointsto main.theS
	return ""
}

func main() {
	print(os.Args) // @pointsto <command-line args>
	fmt.Println("Hello, World!", &theS)
}

// @calls main.main               -> fmt.Println
// @calls (*fmt.pp).handleMethods -> (*main.S).String
