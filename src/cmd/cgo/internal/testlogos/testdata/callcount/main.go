package main

/*
#cgo noescape add
#cgo nocallback add
static int add(int a, int b) { return a + b; }
*/
import "C"

import (
	"fmt"
	"os"
	"runtime"
)

func main() {
	before := runtime.NumCgoCall()
	value := C.add(40, 2)
	after := runtime.NumCgoCall()
	fmt.Printf("value=%d cgo_calls=%d\n", value, after-before)
	if value != 42 || after != before+1 {
		os.Exit(1)
	}
}
