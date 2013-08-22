// +build ignore

package main

import "fmt"

type S int

var theS S

func (s *S) String() string {
	print(s) // @pointsto main.theS
	return ""
}

func main() {
	fmt.Println("Hello, World!", &theS)
}

// @calls main.main               -> fmt.Println
// @calls (*fmt.pp).handleMethods -> (*main.S).String
