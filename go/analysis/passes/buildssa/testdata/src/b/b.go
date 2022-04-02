// Package b contains declarations of generic functions.
package b

import "unsafe"

type Pointer[T any] struct {
	v unsafe.Pointer
}

func (x *Pointer[T]) Load() *T {
	return (*T)(LoadPointer(&x.v))
}

func Load[T any](x *Pointer[T]) *T {
	return x.Load()
}

func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)

// TODO(taking): Add calls from non-generic functions to instantiations of generic functions.
// TODO(taking): Add globals with types that are instantiations of generic functions.
