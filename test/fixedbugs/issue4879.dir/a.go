package a

import (
	"unsafe"
)

type Collection struct {
	root unsafe.Pointer
}

type nodeLoc struct{}

type slice []int

type maptype map[int]int

func MakePrivateCollection() *Collection {
	return &Collection{
		root: unsafe.Pointer(&nodeLoc{}),
	}
}

func MakePrivateCollection2() *Collection {
	return &Collection{
		root: unsafe.Pointer(&slice{}),
	}
}
func MakePrivateCollection3() *Collection {
	return &Collection{
		root: unsafe.Pointer(&maptype{}),
	}
}

