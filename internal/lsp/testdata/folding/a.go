package folding //@fold("package")

import (
	_ "fmt"
	_ "log"
)

import _ "os"

// bar is a function.
// With a multiline doc comment.
func bar() string {
	return `
this string
is not indented`

}
