package folding //@fold("package")

import (
	"fmt"
	_ "log"
)

import _ "os"

// bar is a function.
// With a multiline doc comment.
func bar() string {
	switch {
	case true:
		if true {
			fmt.Println("true")
		} else {
			fmt.Println("false")
		}
	case false:
		fmt.Println("false")
	default:
		fmt.Println("default")
	}
	// This is a multiline comment
	// that is not a doc comment.
	return `
this string
is not indented`
}
