package main

import "C"
import (
	"fmt"
	"os"
	"runtime"
)

//export GoAdd
func GoAdd(a, b C.int) C.int {
	return a + b
}

//export GoGetenv
func GoGetenv(key *C.char) *C.char {
	val := os.Getenv(C.GoString(key))
	return C.CString(val)
}

//export GoRuntimeInfo
func GoRuntimeInfo() *C.char {
	info := fmt.Sprintf("GOOS=%s GOARCH=%s NumCPU=%d", runtime.GOOS, runtime.GOARCH, runtime.NumCPU())
	return C.CString(info)
}

func main() {}
