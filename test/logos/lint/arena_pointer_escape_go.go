// errorcheck

package p

import (
	"reflect"
	"unsafe"
)

type address uintptr
type rawPointer = unsafe.Pointer

func allocatedPointer() unsafe.Pointer {
	return unsafe.Pointer(new(int))
}

func localPointer() unsafe.Pointer {
	var x int
	return unsafe.Pointer(&x)
}

func integerAddress() uintptr {
	var x int
	u := uintptr(unsafe.Pointer(&x))
	return u
}

func immediate(p unsafe.Pointer, offset uintptr) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + offset)
}

func offsetFirst(p unsafe.Pointer, offset uintptr) unsafe.Pointer {
	return unsafe.Pointer(offset + uintptr(p))
}

func masked(p unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) &^ 7)
}

func savedAddress(p unsafe.Pointer) unsafe.Pointer {
	u := uintptr(p)
	return unsafe.Pointer(u) // ERROR "arena_pointer_escape"
}

func arbitraryAddress(u uintptr) unsafe.Pointer {
	return unsafe.Pointer(u) // ERROR "arena_pointer_escape"
}

func aliasAddress(u address) rawPointer {
	return rawPointer(u) // ERROR "arena_pointer_escape"
}

func twoPointers(p, q unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(q)) // ERROR "arena_pointer_escape"
}

func hiddenPointer(p, q unsafe.Pointer) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + uintptr(uint64(uintptr(q)))) // ERROR "arena_pointer_escape"
}

func reflection(v reflect.Value) unsafe.Pointer {
	return unsafe.Pointer(v.Pointer())
}

func reflectionMethod(v reflect.Value) unsafe.Pointer {
	return unsafe.Pointer(reflect.Value.UnsafeAddr(v))
}

func savedReflection(v reflect.Value) unsafe.Pointer {
	u := v.Pointer()
	return unsafe.Pointer(u) // ERROR "arena_pointer_escape"
}

func fabricatedHeader() *reflect.SliceHeader {
	return &reflect.SliceHeader{} // ERROR "arena_pointer_escape"
}

func copiedHeader(h *reflect.StringHeader) reflect.StringHeader {
	return *h // ERROR "arena_pointer_escape"
}

func headerValue(h reflect.SliceHeader) unsafe.Pointer {
	return unsafe.Pointer(h.Data) // ERROR "arena_pointer_escape"
}

func headerOverlay(s []int) unsafe.Pointer {
	h := (*reflect.SliceHeader)(unsafe.Pointer(&s))
	return unsafe.Pointer(h.Data)
}

//logos:allow(correctness::arena_pointer_escape)
func allowedAddress(u uintptr) unsafe.Pointer {
	return unsafe.Pointer(u)
}
