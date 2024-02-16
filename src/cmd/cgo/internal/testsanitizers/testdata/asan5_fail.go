package main

import (
	"fmt"
	"runtime"
	"unsafe"
)

func main() {
	p := new([1024 * 1000]int)
	p[0] = 10
	r := bar(&p[1024*1000-1])
	fmt.Printf("r value is %d", r)
}

func bar(a *int) int {
	p := unsafe.Add(unsafe.Pointer(a), 2*unsafe.Sizeof(int(1)))
	runtime.ASanWrite(p, 8) // BOOM
	*((*int)(p)) = 10
	return *((*int)(p))
}
