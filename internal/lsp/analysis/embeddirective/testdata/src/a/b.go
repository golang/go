package a

import (
	_ "embed"
	"fmt"
)

//go:embed embedText // ok
var s string

// This is main function
func main() {
	fmt.Println(s)
}
