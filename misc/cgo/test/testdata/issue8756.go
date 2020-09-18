package cgotest

/*
#cgo LDFLAGS: -lm
#include <math.h>
*/
import "C"
import (
	"testing"

	"cgotest/issue8756"
)

func test8756(t *testing.T) {
	issue8756.Pow()
	C.pow(1, 2)
}
