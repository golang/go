package main

import "C"

import "unsafe"

//export LLVMFuzzerTestOneInput
func LLVMFuzzerTestOneInput(p unsafe.Pointer, sz C.int) C.int {
	b := C.GoBytes(p, sz)
	if len(b) >= 6 && b[0] == 'F' && b[1] == 'u' && b[2] == 'z' && b[3] == 'z' && b[4] == 'M' && b[5] == 'e' {
		panic("found it")
	}
	return 0
}

func main() {}
