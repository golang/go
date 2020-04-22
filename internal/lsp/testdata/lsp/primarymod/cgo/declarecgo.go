package cgo

/*
#include <stdio.h>
#include <stdlib.h>

void myprint(char* s) {
	printf("%s\n", s);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

func Example() { //@mark(funccgoexample, "Example"),item(funccgoexample, "Example", "func()", "func")
	fmt.Println()
	cs := C.CString("Hello from stdio\n")
	C.myprint(cs)
	C.free(unsafe.Pointer(cs))
}

func _() {
	Example() //@godef("ample", funccgoexample),complete("ample", funccgoexample)
}
