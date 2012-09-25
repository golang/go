// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// MakeFunc implementation.

package reflect

import (
	"runtime"
	"unsafe"
)

// makeFuncImpl is the closure value implementing the function
// returned by MakeFunc.
type makeFuncImpl struct {
	// References visible to the garbage collector.
	// The code array below contains the same references
	// embedded in the machine code.
	typ *commonType
	fn  func([]Value) []Value

	// code is the actual machine code invoked for the closure.
	code [40]byte
}

// MakeFunc returns a new function of the given Type
// that wraps the function fn. When called, that new function
// does the following:
//
//	- converts its arguments to a list of Values args.
//	- runs results := fn(args).
//	- returns the results as a slice of Values, one per formal result.
//
// The implementation fn can assume that the argument Value slice
// has the number and type of arguments given by typ.
// If typ describes a variadic function, the final Value is itself
// a slice representing the variadic arguments, as in the
// body of a variadic function. The result Value slice returned by fn
// must have the number and type of results given by typ.
//
// The Value.Call method allows the caller to invoke a typed function
// in terms of Values; in contrast, MakeFunc allows the caller to implement
// a typed function in terms of Values.
//
// The Examples section of the documentation includes an illustration
// of how to use MakeFunc to build a swap function for different types.
//
func MakeFunc(typ Type, fn func(args []Value) (results []Value)) Value {
	if typ.Kind() != Func {
		panic("reflect: call of MakeFunc with non-Func type")
	}

	// Gather type pointer and function pointers
	// for use in hand-assembled closure.
	t := typ.common()

	// Create function impl.
	// We don't need to save a pointer to makeFuncStub, because it is in
	// the text segment and cannot be garbage collected.
	impl := &makeFuncImpl{
		typ: t,
		fn:  fn,
	}

	tptr := unsafe.Pointer(t)
	fptr := *(*unsafe.Pointer)(unsafe.Pointer(&fn))
	tmp := makeFuncStub
	stub := *(*unsafe.Pointer)(unsafe.Pointer(&tmp))

	// Create code. Copy template and fill in pointer values.
	switch runtime.GOARCH {
	default:
		panic("reflect.MakeFunc: unexpected GOARCH: " + runtime.GOARCH)

	case "amd64":
		copy(impl.code[:], amd64CallStub)
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[2])) = tptr
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[12])) = fptr
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[22])) = stub

	case "386":
		copy(impl.code[:], _386CallStub)
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[1])) = tptr
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[6])) = fptr
		*(*unsafe.Pointer)(unsafe.Pointer(&impl.code[11])) = stub

	case "arm":
		code := (*[10]uintptr)(unsafe.Pointer(&impl.code[0]))
		copy(code[:], armCallStub)
		code[len(armCallStub)] = uintptr(tptr)
		code[len(armCallStub)+1] = uintptr(fptr)
		code[len(armCallStub)+2] = uintptr(stub)

		cacheflush(&impl.code[0], &impl.code[len(impl.code)-1])
	}

	return Value{t, unsafe.Pointer(&impl.code[0]), flag(Func) << flagKindShift}
}

func cacheflush(start, end *byte)

// makeFuncStub is an assembly function used by the code generated
// and returned from MakeFunc. The code returned from makeFunc
// does, schematically,
//
//	MOV $typ, R0
//	MOV $fn, R1
//	MOV $0(FP), R2
//	JMP makeFuncStub
//
// That is, it copies the type and function pointer passed to MakeFunc
// into the first two machine registers and then copies the argument frame
// pointer into the third. Then it jumps to makeFuncStub, which calls callReflect
// with those arguments. Using a jmp to makeFuncStub instead of making the
// call directly keeps the allocated code simpler but, perhaps more
// importantly, also keeps the allocated PCs off the call stack.
// Nothing ever returns to the allocated code.
func makeFuncStub()

// amd64CallStub is the MakeFunc code template for amd64 machines.
var amd64CallStub = []byte{
	// MOVQ $constant, AX
	0x48, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	// MOVQ $constant, BX
	0x48, 0xbb, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	// MOVQ $constant, DX
	0x48, 0xba, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
	// LEAQ 8(SP), CX (argument frame)
	0x48, 0x8d, 0x4c, 0x24, 0x08,
	// JMP *DX
	0xff, 0xe2,
}

// _386CallStub is the MakeFunc code template for 386 machines.
var _386CallStub = []byte{
	// MOVL $constant, AX
	0xb8, 0x00, 0x00, 0x00, 0x00,
	// MOVL $constant, BX
	0xbb, 0x00, 0x00, 0x00, 0x00,
	// MOVL $constant, DX
	0xba, 0x00, 0x00, 0x00, 0x00,
	// LEAL 4(SP), CX (argument frame)
	0x8d, 0x4c, 0x24, 0x04,
	// JMP *DX
	0xff, 0xe2,
}

// armCallStub is the MakeFunc code template for arm machines.
var armCallStub = []uintptr{
	0xe59f000c, // MOVW 0x14(PC), R0
	0xe59f100c, // MOVW 0x14(PC), R1
	0xe28d2004, // MOVW $4(SP), R2
	0xe59ff008, // MOVW 0x10(PC), PC
	0xeafffffe, // B 0(PC), just in case
}
