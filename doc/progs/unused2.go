// compile

package main

import (
	"fmt"
	"io"
)

var _ = fmt.Printf
var _ io.Reader

func main() {
	greeting := "hello, world"
	_ = greeting
}
