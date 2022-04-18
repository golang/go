package a

import (
	"fmt"
)

//go:embed embedText // want "The \"embed\" package must be imported when using go:embed directives"
var s string

// This is main function
func main() {
	fmt.Println(s)
}
