// run
package main

import (
	"fmt"
	"os"
)

func main() {
	w, h := 1<<32, 1<<32
	buf := make([]byte, w*h)
	if len(buf) != 0 {
		fmt.Printf("FAIL overflowing-length-wraps-to-zero: got %d want 0\n", len(buf))
		os.Exit(1)
	}
	fmt.Println("ok overflowing-length-wraps-to-zero")
}
