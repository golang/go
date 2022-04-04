package atomic

import "unsafe"

func LoadPointer(addr *unsafe.Pointer) (val unsafe.Pointer)
