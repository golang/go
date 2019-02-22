package issue8756

/*
#cgo LDFLAGS: -lm
#include <math.h>
*/
import "C"

func Pow() {
	C.pow(1, 2)
}
