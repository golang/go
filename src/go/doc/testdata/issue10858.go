package issue10858

import "unsafe"

// Should be ignored

// First line
//
// Second line
type Type interface {
	// Should be present

	// Align returns the alignment in bytes of a value of
	// this type when allocated in memory.
	Align() int

	// FieldAlign returns the alignment in bytes of a value of
	// this type when used as a field in a struct.
	FieldAlign() int // adjacent comment

	//	Ptr: Elem
	//	Slice: Elem

	// Bits returns the size of the type in bits.

	//
	// It panics if the type's Kind is not one of the
	// sized or unsized Int, Uint, Float, or Complex kinds.
	Bits() int

	// Should be ignored
}

// Should be ignored

// NewType is a comment
//
// ending with this line.
func NewType() Type {}

// Ignore

// First line
//
// Second line
const (
	// Should be ignored

	// C1 comment
	C1 int = 1 << 0

	// Should
	//
	// be ignored

	C2 int = 1 << 1

	// C3 comment
	//
	// with a line gap
	C3 int = 1 << 2

	// Should be ignored
)

// Should be ignored

// Should be ignored

// TypeAlg is a
// copy of runtime.typeAlg
type TypeAlg struct {
	// function for hashing objects of this type
	//
	//
	// (ptr to object, seed) -> hash
	Hash func(unsafe.Pointer, uintptr) uintptr

	// include
	// include

	// include

	// function for comparing objects of this type
	// (ptr to object A, ptr to object B) -> ==?
	Equal func(unsafe.Pointer, unsafe.Pointer) bool
	// Should be ignored
}

// Should be ignored

// StructTag is a comment
//
//
// with 2 connecting lines
type StructTag string // adjacent comment

// Should be ignored

// Get returns the value associated with key in the tag string.
func (tag StructTag) Get(key string) string {
}
