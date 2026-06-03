// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package abi

// These functions are the build-time version of the Go type data structures.

// Their contents must be kept in sync with their definitions.
// Because the host and target type sizes can differ, the compiler and
// linker cannot use the host information that they might get from
// either unsafe.Sizeof and Alignof, nor runtime, reflect, or reflectlite.

// CommonSize returns sizeof(Type) for a compilation target with a given ptrSize
func CommonSize(ptrSize int) int { return 4*ptrSize + 8 + 8 }

// StructFieldSize returns sizeof(StructField) for a compilation target with a given ptrSize
func StructFieldSize(ptrSize int) int { return 3 * ptrSize }

// UncommonSize returns sizeof(UncommonType).  This currently does not depend on ptrSize.
// This exported function is in an internal package, so it may change to depend on ptrSize in the future.
func UncommonSize() uint64 { return 4 + 2 + 2 + 4 + 4 }

// TFlagOff returns the offset of Type.TFlag for a compilation target with a given ptrSize
func TFlagOff(ptrSize int) int { return 2*ptrSize + 4 }

// ITabTypeOff returns the offset of ITab.Type for a compilation target with a given ptrSize
func ITabTypeOff(ptrSize int) int { return ptrSize }

// RTypeSize returns sizeof(kindType) for a compilation target with a given ptrSize.
func RTypeSize(kind Kind, ptrSize int) int {
	cs := CommonSize(ptrSize)
	switch kind {
	case Struct: // reflect.structType
		return cs + 4*ptrSize
	case Pointer: // reflect.ptrType
		return cs + ptrSize
	case Func: // reflect.funcType
		return cs + ptrSize // 4 bytes, pointer aligned
	case Slice: // reflect.sliceType
		return cs + ptrSize
	case Array: // reflect.arrayType
		return cs + 3*ptrSize
	case Chan: // reflect.chanType
		return cs + 2*ptrSize
	case Map: // internal/abi.MapType
		sz := cs + 10*ptrSize + 4
		if ptrSize == 8 {
			sz += 4 // padding for final uint32 field (Flags).
		}
		return sz
	case Interface: // reflect.interfaceType
		return cs + 4*ptrSize
	default:
		// just Sizeof(rtype)
		return cs
	}
}
