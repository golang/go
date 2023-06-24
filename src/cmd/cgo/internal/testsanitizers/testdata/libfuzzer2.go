package main

import "C"

import "unsafe"

//export FuzzMe
func FuzzMe(p unsafe.Pointer, sz C.int) {
	b := C.GoBytes(p, sz)
	b = b[3:]
	if len(b) >= 4 && b[0] == 'f' && b[1] == 'u' && b[2] == 'z' && b[3] == 'z' {
		panic("found it")
	}
}

func main() {}
