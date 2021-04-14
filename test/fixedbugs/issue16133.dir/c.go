package p

import (
	"./a1"
	"./b"
)

var _ = b.T{
	X: a.NewX(), // ERROR `cannot use "a1"\.NewX\(\)`
}
