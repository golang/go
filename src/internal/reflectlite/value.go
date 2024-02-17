// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflectlite

import (
	"internal/abi"
	"internal/goarch"
	"internal/unsafeheader"
	"runtime"
	"unsafe"
)

// Value is the reflection interface to a Go value.
//
// Not all methods apply to all kinds of values. Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of value before
// calling kind-specific methods. Calling a method
// inappropriate to the kind of type causes a run time panic.
//
// The zero Value represents no value.
// Its IsValid method returns false, its Kind method returns Invalid,
// its String method returns "<invalid Value>", and all other methods panic.
// Most functions and methods never return an invalid value.
// If one does, its documentation states the conditions explicitly.
//
// A Value can be used concurrently by multiple goroutines provided that
// the underlying Go value can be used concurrently for the equivalent
// direct operations.
//
// To compare two Values, compare the results of the Interface method.
// Using == on two Values does not compare the underlying values
// they represent.
type Value struct {
	// typ_ holds the type of the value represented by a Value.
	// Access using the typ method to avoid escape of v.
	typ_ *abi.Type

	// Pointer-valued data or, if flagIndir is set, pointer to data.
	// Valid when either flagIndir is set or typ.pointers() is true.
	ptr unsafe.Pointer

	// flag holds metadata about the value.
	// The lowest bits are flag bits:
	//	- flagStickyRO: obtained via unexported not embedded field, so read-only
	//	- flagEmbedRO: obtained via unexported embedded field, so read-only
	//	- flagIndir: val holds a pointer to the data
	//	- flagAddr: v.CanAddr is true (implies flagIndir)
	// Value cannot represent method values.
	// The next five bits give the Kind of the value.
	// This repeats typ.Kind() except for method values.
	// The remaining 23+ bits give a method number for method values.
	// If flag.kind() != Func, code can assume that flagMethod is unset.
	// If ifaceIndir(typ), code can assume that flagIndir is set.
	Flag

	// A method value represents a curried method invocation
	// like r.Read for some receiver r. The typ+val+flag bits describe
	// the receiver r, but the flag's Kind bits say Func (methods are
	// functions), and the top bits of the flag give the method number
	// in r's type's method table.
}

func (v Value) typ() *abi.Type {
	// Types are either static (for compiler-created types) or
	// heap-allocated but always reachable (for reflection-created
	// types, held in the central map). So there is no need to
	// escape types. noescape here help avoid unnecessary escape
	// of v.
	return (*abi.Type)(noescape(unsafe.Pointer(v.typ_)))
}

// pointer returns the underlying pointer represented by v.
// v.Kind() must be Pointer, Map, Chan, Func, or UnsafePointer
func (v Value) pointer() unsafe.Pointer {
	if v.typ().Size() != goarch.PtrSize || !v.typ().Pointers() {
		panic("can't call pointer on a non-pointer Value")
	}
	if v.Flag&FlagIndir != 0 {
		return *(*unsafe.Pointer)(v.ptr)
	}
	return v.ptr
}

// packEface converts v to the empty interface.
func packEface(v Value) any {
	t := v.typ()
	var i any
	e := (*emptyInterface)(unsafe.Pointer(&i))
	// First, fill in the data portion of the interface.
	switch {
	case ifaceIndir(t):
		if v.Flag&FlagIndir == 0 {
			panic("bad indir")
		}
		// Value is indirect, and so is the interface we're making.
		ptr := v.ptr
		if v.Flag&FlagAddr != 0 {
			c := unsafe_New(t)
			typedmemmove(t, c, ptr)
			ptr = c
		}
		e.word = ptr
	case v.Flag&FlagIndir != 0:
		// Value is indirect, but interface is direct. We need
		// to load the data at v.ptr into the interface data word.
		e.word = *(*unsafe.Pointer)(v.ptr)
	default:
		// Value is direct, and so is the interface.
		e.word = v.ptr
	}
	// Now, fill in the type portion. We're very careful here not
	// to have any operation between the e.word and e.typ assignments
	// that would let the garbage collector observe the partially-built
	// interface value.
	e.typ = t
	return i
}

// unpackEface converts the empty interface i to a Value.
func unpackEface(i any) Value {
	e := (*emptyInterface)(unsafe.Pointer(&i))
	// NOTE: don't read e.word until we know whether it is really a pointer or not.
	t := e.typ
	if t == nil {
		return Value{}
	}
	f := Flag(t.Kind())
	if ifaceIndir(t) {
		f |= FlagIndir
	}
	return Value{t, e.word, f}
}

// methodName returns the name of the calling method,
// assumed to be two stack frames above.
func methodName() string {
	pc, _, _, _ := runtime.Caller(2)
	f := runtime.FuncForPC(pc)
	if f == nil {
		return "unknown method"
	}
	return f.Name()
}

// emptyInterface is the header for an interface{} value.
type emptyInterface struct {
	typ  *abi.Type
	word unsafe.Pointer
}

// CanSet reports whether the value of v can be changed.
// A Value can be changed only if it is addressable and was not
// obtained by the use of unexported struct fields.
// If CanSet returns false, calling Set or any type-specific
// setter (e.g., SetBool, SetInt) will panic.
func (v Value) CanSet() bool {
	return v.Flag&(FlagAddr|FlagRO) == FlagAddr
}

// Elem returns the value that the interface v contains
// or that the pointer v points to.
// It panics if v's Kind is not Interface or Pointer.
// It returns the zero Value if v is nil.
func (v Value) Elem() Value {
	k := v.Kind()
	switch k {
	case abi.Interface:
		var eface any
		if v.typ().NumMethod() == 0 {
			eface = *(*any)(v.ptr)
		} else {
			eface = (any)(*(*interface {
				M()
			})(v.ptr))
		}
		x := unpackEface(eface)
		if x.Flag != 0 {
			x.Flag |= v.Flag.Ro()
		}
		return x
	case abi.Pointer:
		ptr := v.ptr
		if v.Flag&FlagIndir != 0 {
			ptr = *(*unsafe.Pointer)(ptr)
		}
		// The returned value's address is v's value.
		if ptr == nil {
			return Value{}
		}
		tt := (*ptrType)(unsafe.Pointer(v.typ()))
		typ := tt.Elem
		fl := v.Flag&FlagRO | FlagIndir | FlagAddr
		fl |= Flag(typ.Kind())
		return Value{typ, ptr, fl}
	}
	panic(&ValueError{"reflectlite.Value.Elem", v.Kind()})
}

func valueInterface(v Value) any {
	if v.Flag == 0 {
		panic(&ValueError{"reflectlite.Value.Interface", 0})
	}

	if v.Kind() == abi.Interface {
		// Special case: return the element inside the interface.
		// Empty interface has one layout, all interfaces with
		// methods have a second layout.
		if v.numMethod() == 0 {
			return *(*any)(v.ptr)
		}
		return *(*interface {
			M()
		})(v.ptr)
	}

	return packEface(v)
}

// IsNil reports whether its argument v is nil. The argument must be
// a chan, func, interface, map, pointer, or slice value; if it is
// not, IsNil panics. Note that IsNil is not always equivalent to a
// regular comparison with nil in Go. For example, if v was created
// by calling ValueOf with an uninitialized interface variable i,
// i==nil will be true but v.IsNil will panic as v will be the zero
// Value.
func (v Value) IsNil() bool {
	k := v.Kind()
	switch k {
	case abi.Chan, abi.Func, abi.Map, abi.Pointer, abi.UnsafePointer:
		// if v.flag&flagMethod != 0 {
		// 	return false
		// }
		ptr := v.ptr
		if v.Flag&FlagIndir != 0 {
			ptr = *(*unsafe.Pointer)(ptr)
		}
		return ptr == nil
	case abi.Interface, abi.Slice:
		// Both interface and slice are nil if first word is 0.
		// Both are always bigger than a word; assume flagIndir.
		return *(*unsafe.Pointer)(v.ptr) == nil
	}
	panic(&ValueError{"reflectlite.Value.IsNil", v.Kind()})
}

// IsValid reports whether v represents a value.
// It returns false if v is the zero Value.
// If IsValid returns false, all other methods except String panic.
// Most functions and methods never return an invalid Value.
// If one does, its documentation states the conditions explicitly.
func (v Value) IsValid() bool {
	return v.Flag != 0
}

// Kind returns v's Kind.
// If v is the zero Value (IsValid returns false), Kind returns Invalid.
func (v Value) Kind() Kind {
	return v.Flag.Kind()
}

// implemented in runtime:

//go:noescape
func chanlen(unsafe.Pointer) int

//go:noescape
func maplen(unsafe.Pointer) int

// Len returns v's length.
// It panics if v's Kind is not Array, Chan, Map, Slice, or String.
func (v Value) Len() int {
	k := v.Kind()
	switch k {
	case abi.Array:
		tt := (*arrayType)(unsafe.Pointer(v.typ()))
		return int(tt.Len)
	case abi.Chan:
		return chanlen(v.pointer())
	case abi.Map:
		return maplen(v.pointer())
	case abi.Slice:
		// Slice is bigger than a word; assume flagIndir.
		return (*unsafeheader.Slice)(v.ptr).Len
	case abi.String:
		// String is bigger than a word; assume flagIndir.
		return (*unsafeheader.String)(v.ptr).Len
	}
	panic(&ValueError{"reflect.Value.Len", v.Kind()})
}

// NumMethod returns the number of exported methods in the value's method set.
func (v Value) numMethod() int {
	if v.typ() == nil {
		panic(&ValueError{"reflectlite.Value.NumMethod", abi.Invalid})
	}
	return v.typ().NumMethod()
}

// Set assigns x to the value v.
// It panics if CanSet returns false.
// As in Go, x's value must be assignable to v's type.
func (v Value) Set(x Value) {
	v.MustBeAssignable()
	x.MustBeExported() // do not let unexported x leak
	var target unsafe.Pointer
	if v.Kind() == abi.Interface {
		target = v.ptr
	}
	x = x.assignTo("reflectlite.Set", v.typ(), target)
	if x.Flag&FlagIndir != 0 {
		typedmemmove(v.typ(), v.ptr, x.ptr)
	} else {
		*(*unsafe.Pointer)(v.ptr) = x.ptr
	}
}

// Type returns v's type.
func (v Value) Type() Type {
	f := v.Flag
	if f == 0 {
		panic(&ValueError{"reflectlite.Value.Type", abi.Invalid})
	}
	// Method values not supported.
	return toRType(v.typ())
}

/*
 * constructors
 */

// implemented in package runtime

//go:noescape
func unsafe_New(*abi.Type) unsafe.Pointer

// ValueOf returns a new Value initialized to the concrete value
// stored in the interface i. ValueOf(nil) returns the zero Value.
func ValueOf(i any) Value {
	if i == nil {
		return Value{}
	}
	return unpackEface(i)
}

// assignTo returns a value v that can be assigned directly to typ.
// It panics if v is not assignable to typ.
// For a conversion to an interface type, target is a suggested scratch space to use.
func (v Value) assignTo(context string, dst *abi.Type, target unsafe.Pointer) Value {
	// if v.flag&flagMethod != 0 {
	// 	v = makeMethodValue(context, v)
	// }

	switch {
	case directlyAssignable(dst, v.typ()):
		// Overwrite type so that they match.
		// Same memory layout, so no harm done.
		fl := v.Flag&(FlagAddr|FlagIndir) | v.Flag.Ro()
		fl |= Flag(dst.Kind())
		return Value{dst, v.ptr, fl}

	case implements(dst, v.typ()):
		if target == nil {
			target = unsafe_New(dst)
		}
		if v.Kind() == abi.Interface && v.IsNil() {
			// A nil ReadWriter passed to nil Reader is OK,
			// but using ifaceE2I below will panic.
			// Avoid the panic by returning a nil dst (e.g., Reader) explicitly.
			return Value{dst, nil, Flag(abi.Interface)}
		}
		x := valueInterface(v)
		if dst.NumMethod() == 0 {
			*(*any)(target) = x
		} else {
			ifaceE2I(dst, x, target)
		}
		return Value{dst, target, FlagIndir | Flag(Interface)}
	}

	// Failed.
	panic(context + ": value of type " + toRType(v.typ()).String() + " is not assignable to type " + toRType(dst).String())
}

// arrayAt returns the i-th element of p,
// an array whose elements are eltSize bytes wide.
// The array pointed at by p must have at least i+1 elements:
// it is invalid (but impossible to check here) to pass i >= len,
// because then the result will point outside the array.
// whySafe must explain why i < len. (Passing "i < len" is fine;
// the benefit is to surface this assumption at the call site.)
func arrayAt(p unsafe.Pointer, i int, eltSize uintptr, whySafe string) unsafe.Pointer {
	return add(p, uintptr(i)*eltSize, "i < len")
}

func ifaceE2I(t *abi.Type, src any, dst unsafe.Pointer)

// typedmemmove copies a value of type t to dst from src.
//
//go:noescape
func typedmemmove(t *abi.Type, dst, src unsafe.Pointer)

// Dummy annotation marking that the value x escapes,
// for use in cases where the reflect code is so clever that
// the compiler cannot follow.
func escapes(x any) {
	if dummy.b {
		dummy.x = x
	}
}

var dummy struct {
	b bool
	x any
}

//go:nosplit
func noescape(p unsafe.Pointer) unsafe.Pointer {
	x := uintptr(p)
	return unsafe.Pointer(x ^ 0)
}

type Flag uintptr

const (
	FlagKindWidth   Flag = 5 // there are 27 kinds
	FlagKindMask    Flag = 1<<FlagKindWidth - 1
	FlagStickyRO    Flag = 1 << 5
	FlagEmbedRO     Flag = 1 << 6
	FlagIndir       Flag = 1 << 7
	FlagAddr        Flag = 1 << 8
	FlagMethod      Flag = 1 << 9
	FlagMethodShift Flag = 10
	FlagRO          Flag = FlagStickyRO | FlagEmbedRO
)

func (f Flag) Kind() Kind {
	return Kind(f & FlagKindMask)
}

func (f Flag) Ro() Flag {
	if f&FlagRO != 0 {
		return FlagStickyRO
	}
	return 0
}

// mustBe panics if f's kind is not expected.
// Making this a method on flag instead of on  [reflect.Value]
// (and embedding flag in Value) means that we can write
// the very clear v.mustBe(Bool) and have it compile into
// v.flag.mustBe(Bool), which will only bother to copy the
// single important word for the receiver.
func (f Flag) MustBe(expected Kind) {
	if f.Kind() != expected {
		mustBePanic(f.Kind())
	}
}

//go:noinline
func mustBePanic(k Kind) {
	panic(NewValueError(valueMethodName(), k))
}

// mustBeExported panics if f records that the value was obtained using
// an unexported field.
func (f Flag) MustBeExported() {
	if f == 0 || f&FlagRO != 0 {
		f.MustBeExportedSlow()
	}
}

func (f Flag) MustBeExportedSlow() {
	if f == 0 {
		panic(NewValueError(valueMethodName(), abi.Invalid))
	}
	if f&FlagRO != 0 {
		panic("reflect: " + valueMethodName() + " using value obtained using unexported field")
	}
}

// mustBeAssignable panics if f records that the value is not assignable,
// which is to say that either it was obtained using an unexported field
// or it is not addressable.
func (f Flag) MustBeAssignable() {
	if f&FlagRO != 0 || f&FlagAddr == 0 {
		f.MustBeAssignableSlow()
	}
}

func (f Flag) MustBeAssignableSlow() {
	if f == 0 {
		panic(NewValueError(valueMethodName(), abi.Invalid))
	}
	// Assignable if addressable and not read-only.
	if f&FlagRO != 0 {
		panic("reflect: " + valueMethodName() + " using value obtained using unexported field")
	}
	if f&FlagAddr == 0 {
		panic("reflect: " + valueMethodName() + " using unaddressable value")
	}
}

// Force slow panicking path not inlined, so it won't add to the
// inlining budget of the caller.
// TODO: undo when the inliner is no longer bottom-up only.
//
//go:noinline
func (f Flag) PanicNotMap() {
	f.MustBe(abi.Map)
}

// A ValueError occurs when a Value method is invoked on
// a [Value] that does not support it. Such cases are documented
// in the description of each method.
type ValueError struct {
	Method string
	Kind   Kind
}

func (e *ValueError) Error() string {
	if e.Kind == 0 {
		return "reflect: call of " + e.Method + " on zero Value"
	}
	return "reflect: call of " + e.Method + " on " + e.Kind.String() + " Value"
}

// valueMethodName returns the name of the exported calling method on Value.
func valueMethodName() string {
	var pc [5]uintptr
	n := runtime.Callers(1,
		(*(*[5]uintptr)(noescape(unsafe.Pointer(&pc[0]))))[:],
	)
	frames := runtime.CallersFrames((*(*[5]uintptr)(noescape(unsafe.Pointer(&pc[0]))))[:n])
	var frame runtime.Frame
	for more := true; more; {
		const prefix = "reflect.Value."
		frame, more = frames.Next()
		name := frame.Function
		if len(name) > len(prefix) && name[:len(prefix)] == prefix {
			methodName := name[len(prefix):]
			if len(methodName) > 0 && 'A' <= methodName[0] && methodName[0] <= 'Z' {
				return name
			}
		}
	}
	return "unknown method"
}

// NewValueError default return [ValueError].
// When reflect is imported, return [reflect.ValueError].
var NewValueError func(Method string, Kind abi.Kind) interface{} = func(Method string, Kind abi.Kind) interface{} {
	return &ValueError{Method: Method, Kind: Kind}
}
