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
		}
	case false:
		fmt.Println("false")
	default:
		fmt.Println("default")
	}

	return `
this string
is not indented`

}
