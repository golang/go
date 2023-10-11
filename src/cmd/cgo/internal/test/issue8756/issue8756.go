package issue8756

/*
#cgo !darwin LDFLAGS: -lm
#include <math.h>
*/
import "C"

func Pow() {
	C.pow(1, 2)
}
