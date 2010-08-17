// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"math"
	"runtime"
	"unsafe"
)

const ptrSize = uintptr(unsafe.Sizeof((*byte)(nil)))
const cannotSet = "cannot set value obtained via unexported struct field"

type addr unsafe.Pointer

// TODO: This will have to go away when
// the new gc goes in.
func memmove(adst, asrc addr, n uintptr) {
	dst := uintptr(adst)
	src := uintptr(asrc)
	switch {
	case src < dst && src+n > dst:
		// byte copy backward
		// careful: i is unsigned
		for i := n; i > 0; {
			i--
			*(*byte)(addr(dst + i)) = *(*byte)(addr(src + i))
		}
	case (n|src|dst)&(ptrSize-1) != 0:
		// byte copy forward
		for i := uintptr(0); i < n; i++ {
			*(*byte)(addr(dst + i)) = *(*byte)(addr(src + i))
		}
	default:
		// word copy forward
		for i := uintptr(0); i < n; i += ptrSize {
			*(*uintptr)(addr(dst + i)) = *(*uintptr)(addr(src + i))
		}
	}
}

// Value is the common interface to reflection values.
// The implementations of Value (e.g., ArrayValue, StructValue)
// have additional type-specific methods.
type Value interface {
	// Type returns the value's type.
	Type() Type

	// Interface returns the value as an interface{}.
	Interface() interface{}

	// CanSet returns whether the value can be changed.
	// Values obtained by the use of non-exported struct fields
	// can be used in Get but not Set.
	// If CanSet() returns false, calling the type-specific Set
	// will cause a crash.
	CanSet() bool

	// SetValue assigns v to the value; v must have the same type as the value.
	SetValue(v Value)

	// Addr returns a pointer to the underlying data.
	// It is for advanced clients that also
	// import the "unsafe" package.
	Addr() uintptr

	// Method returns a FuncValue corresponding to the value's i'th method.
	// The arguments to a Call on the returned FuncValue
	// should not include a receiver; the FuncValue will use
	// the value as the receiver.
	Method(i int) *FuncValue

	getAddr() addr
}

// value is the common implementation of most values.
// It is embedded in other, public struct types, but always
// with a unique tag like "uint" or "float" so that the client cannot
// convert from, say, *UintValue to *FloatValue.
type value struct {
	typ    Type
	addr   addr
	canSet bool
}

func (v *value) Type() Type { return v.typ }

func (v *value) Addr() uintptr { return uintptr(v.addr) }

func (v *value) getAddr() addr { return v.addr }

func (v *value) Interface() interface{} {
	if typ, ok := v.typ.(*InterfaceType); ok {
		// There are two different representations of interface values,
		// one if the interface type has methods and one if it doesn't.
		// These two representations require different expressions
		// to extract correctly.
		if typ.NumMethod() == 0 {
			// Extract as interface value without methods.
			return *(*interface{})(v.addr)
		}
		// Extract from v.addr as interface value with methods.
		return *(*interface {
			m()
		})(v.addr)
	}
	return unsafe.Unreflect(v.typ, unsafe.Pointer(v.addr))
}

func (v *value) CanSet() bool { return v.canSet }

/*
 * basic types
 */

// BoolValue represents a bool value.
type BoolValue struct {
	value "bool"
}

// Get returns the underlying bool value.
func (v *BoolValue) Get() bool { return *(*bool)(v.addr) }

// Set sets v to the value x.
func (v *BoolValue) Set(x bool) {
	if !v.canSet {
		panic(cannotSet)
	}
	*(*bool)(v.addr) = x
}

// Set sets v to the value x.
func (v *BoolValue) SetValue(x Value) { v.Set(x.(*BoolValue).Get()) }

// FloatValue represents a float value.
type FloatValue struct {
	value "float"
}

// Get returns the underlying int value.
func (v *FloatValue) Get() float64 {
	switch v.typ.(*FloatType).Kind() {
	case Float:
		return float64(*(*float)(v.addr))
	case Float32:
		return float64(*(*float32)(v.addr))
	case Float64:
		return *(*float64)(v.addr)
	}
	panic("reflect: invalid float kind")
}

// Set sets v to the value x.
func (v *FloatValue) Set(x float64) {
	if !v.canSet {
		panic(cannotSet)
	}
	switch v.typ.(*FloatType).Kind() {
	default:
		panic("reflect: invalid float kind")
	case Float:
		*(*float)(v.addr) = float(x)
	case Float32:
		*(*float32)(v.addr) = float32(x)
	case Float64:
		*(*float64)(v.addr) = x
	}
}

// Overflow returns true if x cannot be represented by the type of v.
func (v *FloatValue) Overflow(x float64) bool {
	if v.typ.Size() == 8 {
		return false
	}
	if x < 0 {
		x = -x
	}
	return math.MaxFloat32 < x && x <= math.MaxFloat64
}

// Set sets v to the value x.
func (v *FloatValue) SetValue(x Value) { v.Set(x.(*FloatValue).Get()) }

// ComplexValue represents a complex value.
type ComplexValue struct {
	value "complex"
}

// Get returns the underlying complex value.
func (v *ComplexValue) Get() complex128 {
	switch v.typ.(*ComplexType).Kind() {
	case Complex:
		return complex128(*(*complex)(v.addr))
	case Complex64:
		return complex128(*(*complex64)(v.addr))
	case Complex128:
		return *(*complex128)(v.addr)
	}
	panic("reflect: invalid complex kind")
}

// Set sets v to the value x.
func (v *ComplexValue) Set(x complex128) {
	if !v.canSet {
		panic(cannotSet)
	}
	switch v.typ.(*ComplexType).Kind() {
	default:
		panic("reflect: invalid complex kind")
	case Complex:
		*(*complex)(v.addr) = complex(x)
	case Complex64:
		*(*complex64)(v.addr) = complex64(x)
	case Complex128:
		*(*complex128)(v.addr) = x
	}
}

// Set sets v to the value x.
func (v *ComplexValue) SetValue(x Value) { v.Set(x.(*ComplexValue).Get()) }

// IntValue represents an int value.
type IntValue struct {
	value "int"
}

// Get returns the underlying int value.
func (v *IntValue) Get() int64 {
	switch v.typ.(*IntType).Kind() {
	case Int:
		return int64(*(*int)(v.addr))
	case Int8:
		return int64(*(*int8)(v.addr))
	case Int16:
		return int64(*(*int16)(v.addr))
	case Int32:
		return int64(*(*int32)(v.addr))
	case Int64:
		return *(*int64)(v.addr)
	}
	panic("reflect: invalid int kind")
}

// Set sets v to the value x.
func (v *IntValue) Set(x int64) {
	if !v.canSet {
		panic(cannotSet)
	}
	switch v.typ.(*IntType).Kind() {
	default:
		panic("reflect: invalid int kind")
	case Int:
		*(*int)(v.addr) = int(x)
	case Int8:
		*(*int8)(v.addr) = int8(x)
	case Int16:
		*(*int16)(v.addr) = int16(x)
	case Int32:
		*(*int32)(v.addr) = int32(x)
	case Int64:
		*(*int64)(v.addr) = x
	}
}

// Set sets v to the value x.
func (v *IntValue) SetValue(x Value) { v.Set(x.(*IntValue).Get()) }

// Overflow returns true if x cannot be represented by the type of v.
func (v *IntValue) Overflow(x int64) bool {
	bitSize := uint(v.typ.Bits())
	trunc := (x << (64 - bitSize)) >> (64 - bitSize)
	return x != trunc
}

// StringHeader is the runtime representation of a string.
type StringHeader struct {
	Data uintptr
	Len  int
}

// StringValue represents a string value.
type StringValue struct {
	value "string"
}

// Get returns the underlying string value.
func (v *StringValue) Get() string { return *(*string)(v.addr) }

// Set sets v to the value x.
func (v *StringValue) Set(x string) {
	if !v.canSet {
		panic(cannotSet)
	}
	*(*string)(v.addr) = x
}

// Set sets v to the value x.
func (v *StringValue) SetValue(x Value) { v.Set(x.(*StringValue).Get()) }

// UintValue represents a uint value.
type UintValue struct {
	value "uint"
}

// Get returns the underlying uuint value.
func (v *UintValue) Get() uint64 {
	switch v.typ.(*UintType).Kind() {
	case Uint:
		return uint64(*(*uint)(v.addr))
	case Uint8:
		return uint64(*(*uint8)(v.addr))
	case Uint16:
		return uint64(*(*uint16)(v.addr))
	case Uint32:
		return uint64(*(*uint32)(v.addr))
	case Uint64:
		return *(*uint64)(v.addr)
	case Uintptr:
		return uint64(*(*uintptr)(v.addr))
	}
	panic("reflect: invalid uint kind")
}

// Set sets v to the value x.
func (v *UintValue) Set(x uint64) {
	if !v.canSet {
		panic(cannotSet)
	}
	switch v.typ.(*UintType).Kind() {
	default:
		panic("reflect: invalid uint kind")
	case Uint:
		*(*uint)(v.addr) = uint(x)
	case Uint8:
		*(*uint8)(v.addr) = uint8(x)
	case Uint16:
		*(*uint16)(v.addr) = uint16(x)
	case Uint32:
		*(*uint32)(v.addr) = uint32(x)
	case Uint64:
		*(*uint64)(v.addr) = x
	case Uintptr:
		*(*uintptr)(v.addr) = uintptr(x)
	}
}

// Overflow returns true if x cannot be represented by the type of v.
func (v *UintValue) Overflow(x uint64) bool {
	bitSize := uint(v.typ.Bits())
	trunc := (x << (64 - bitSize)) >> (64 - bitSize)
	return x != trunc
}

// Set sets v to the value x.
func (v *UintValue) SetValue(x Value) { v.Set(x.(*UintValue).Get()) }

// UnsafePointerValue represents an unsafe.Pointer value.
type UnsafePointerValue struct {
	value "unsafe.Pointer"
}

// Get returns the underlying uintptr value.
// Get returns uintptr, not unsafe.Pointer, so that
// programs that do not import "unsafe" cannot
// obtain a value of unsafe.Pointer type from "reflect".
func (v *UnsafePointerValue) Get() uintptr { return uintptr(*(*unsafe.Pointer)(v.addr)) }

// Set sets v to the value x.
func (v *UnsafePointerValue) Set(x unsafe.Pointer) {
	if !v.canSet {
		panic(cannotSet)
	}
	*(*unsafe.Pointer)(v.addr) = x
}

// Set sets v to the value x.
func (v *UnsafePointerValue) SetValue(x Value) {
	v.Set(unsafe.Pointer(x.(*UnsafePointerValue).Get()))
}

func typesMustMatch(t1, t2 Type) {
	if t1 != t2 {
		panic("type mismatch: " + t1.String() + " != " + t2.String())
	}
}

/*
 * array
 */

// ArrayOrSliceValue is the common interface
// implemented by both ArrayValue and SliceValue.
type ArrayOrSliceValue interface {
	Value
	Len() int
	Cap() int
	Elem(i int) Value
	addr() addr
}

// ArrayCopy copies the contents of src into dst until either
// dst has been filled or src has been exhausted.
// It returns the number of elements copied.
// The arrays dst and src must have the same element type.
func ArrayCopy(dst, src ArrayOrSliceValue) int {
	// TODO: This will have to move into the runtime
	// once the real gc goes in.
	de := dst.Type().(ArrayOrSliceType).Elem()
	se := src.Type().(ArrayOrSliceType).Elem()
	typesMustMatch(de, se)
	n := dst.Len()
	if xn := src.Len(); n > xn {
		n = xn
	}
	memmove(dst.addr(), src.addr(), uintptr(n)*de.Size())
	return n
}

// An ArrayValue represents an array.
type ArrayValue struct {
	value "array"
}

// Len returns the length of the array.
func (v *ArrayValue) Len() int { return v.typ.(*ArrayType).Len() }

// Cap returns the capacity of the array (equal to Len()).
func (v *ArrayValue) Cap() int { return v.typ.(*ArrayType).Len() }

// addr returns the base address of the data in the array.
func (v *ArrayValue) addr() addr { return v.value.addr }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ArrayValue) Set(x *ArrayValue) {
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	ArrayCopy(v, x)
}

// Set sets v to the value x.
func (v *ArrayValue) SetValue(x Value) { v.Set(x.(*ArrayValue)) }

// Elem returns the i'th element of v.
func (v *ArrayValue) Elem(i int) Value {
	typ := v.typ.(*ArrayType).Elem()
	n := v.Len()
	if i < 0 || i >= n {
		panic("array index out of bounds")
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size())
	return newValue(typ, p, v.canSet)
}

/*
 * slice
 */

// runtime representation of slice
type SliceHeader struct {
	Data uintptr
	Len  int
	Cap  int
}

// A SliceValue represents a slice.
type SliceValue struct {
	value "slice"
}

func (v *SliceValue) slice() *SliceHeader { return (*SliceHeader)(v.value.addr) }

// IsNil returns whether v is a nil slice.
func (v *SliceValue) IsNil() bool { return v.slice().Data == 0 }

// Len returns the length of the slice.
func (v *SliceValue) Len() int { return int(v.slice().Len) }

// Cap returns the capacity of the slice.
func (v *SliceValue) Cap() int { return int(v.slice().Cap) }

// addr returns the base address of the data in the slice.
func (v *SliceValue) addr() addr { return addr(v.slice().Data) }

// SetLen changes the length of v.
// The new length n must be between 0 and the capacity, inclusive.
func (v *SliceValue) SetLen(n int) {
	s := v.slice()
	if n < 0 || n > int(s.Cap) {
		panic("reflect: slice length out of range in SetLen")
	}
	s.Len = n
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *SliceValue) Set(x *SliceValue) {
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*v.slice() = *x.slice()
}

// Set sets v to the value x.
func (v *SliceValue) SetValue(x Value) { v.Set(x.(*SliceValue)) }

// Get returns the uintptr address of the v.Cap()'th element.  This gives
// the same result for all slices of the same array.
// It is mainly useful for printing.
func (v *SliceValue) Get() uintptr {
	typ := v.typ.(*SliceType)
	return uintptr(v.addr()) + uintptr(v.Cap())*typ.Elem().Size()
}

// Slice returns a sub-slice of the slice v.
func (v *SliceValue) Slice(beg, end int) *SliceValue {
	cap := v.Cap()
	if beg < 0 || end < beg || end > cap {
		panic("slice index out of bounds")
	}
	typ := v.typ.(*SliceType)
	s := new(SliceHeader)
	s.Data = uintptr(v.addr()) + uintptr(beg)*typ.Elem().Size()
	s.Len = end - beg
	s.Cap = cap - beg
	return newValue(typ, addr(s), v.canSet).(*SliceValue)
}

// Elem returns the i'th element of v.
func (v *SliceValue) Elem(i int) Value {
	typ := v.typ.(*SliceType).Elem()
	n := v.Len()
	if i < 0 || i >= n {
		panic("reflect: slice index out of range")
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size())
	return newValue(typ, p, v.canSet)
}

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ *SliceType, len, cap int) *SliceValue {
	s := &SliceHeader{
		Data: uintptr(unsafe.NewArray(typ.Elem(), cap)),
		Len:  len,
		Cap:  cap,
	}
	return newValue(typ, addr(s), true).(*SliceValue)
}

/*
 * chan
 */

// A ChanValue represents a chan.
type ChanValue struct {
	value "chan"
}

// IsNil returns whether v is a nil channel.
func (v *ChanValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ChanValue) Set(x *ChanValue) {
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *ChanValue) SetValue(x Value) { v.Set(x.(*ChanValue)) }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *ChanValue) Get() uintptr { return *(*uintptr)(v.addr) }

// implemented in ../pkg/runtime/reflect.cgo
func makechan(typ *runtime.ChanType, size uint32) (ch *byte)
func chansend(ch, val *byte, pres *bool)
func chanrecv(ch, val *byte, pres *bool)
func chanclosed(ch *byte) bool
func chanclose(ch *byte)
func chanlen(ch *byte) int32
func chancap(ch *byte) int32

// Closed returns the result of closed(c) on the underlying channel.
func (v *ChanValue) Closed() bool {
	ch := *(**byte)(v.addr)
	return chanclosed(ch)
}

// Close closes the channel.
func (v *ChanValue) Close() {
	ch := *(**byte)(v.addr)
	chanclose(ch)
}

func (v *ChanValue) Len() int {
	ch := *(**byte)(v.addr)
	return int(chanlen(ch))
}

func (v *ChanValue) Cap() int {
	ch := *(**byte)(v.addr)
	return int(chancap(ch))
}

// internal send; non-blocking if b != nil
func (v *ChanValue) send(x Value, b *bool) {
	t := v.Type().(*ChanType)
	if t.Dir()&SendDir == 0 {
		panic("send on recv-only channel")
	}
	typesMustMatch(t.Elem(), x.Type())
	ch := *(**byte)(v.addr)
	chansend(ch, (*byte)(x.getAddr()), b)
}

// internal recv; non-blocking if b != nil
func (v *ChanValue) recv(b *bool) Value {
	t := v.Type().(*ChanType)
	if t.Dir()&RecvDir == 0 {
		panic("recv on send-only channel")
	}
	ch := *(**byte)(v.addr)
	x := MakeZero(t.Elem())
	chanrecv(ch, (*byte)(x.getAddr()), b)
	return x
}

// Send sends x on the channel v.
func (v *ChanValue) Send(x Value) { v.send(x, nil) }

// Recv receives and returns a value from the channel v.
func (v *ChanValue) Recv() Value { return v.recv(nil) }

// TrySend attempts to sends x on the channel v but will not block.
// It returns true if the value was sent, false otherwise.
func (v *ChanValue) TrySend(x Value) bool {
	var ok bool
	v.send(x, &ok)
	return ok
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It returns the value if one is received, nil otherwise.
func (v *ChanValue) TryRecv() Value {
	var ok bool
	x := v.recv(&ok)
	if !ok {
		return nil
	}
	return x
}

// MakeChan creates a new channel with the specified type and buffer size.
func MakeChan(typ *ChanType, buffer int) *ChanValue {
	if buffer < 0 {
		panic("MakeChan: negative buffer size")
	}
	if typ.Dir() != BothDir {
		panic("MakeChan: unidirectional channel type")
	}
	v := MakeZero(typ).(*ChanValue)
	*(**byte)(v.addr) = makechan((*runtime.ChanType)(unsafe.Pointer(typ)), uint32(buffer))
	return v
}

/*
 * func
 */

// A FuncValue represents a function value.
type FuncValue struct {
	value       "func"
	first       *value
	isInterface bool
}

// IsNil returns whether v is a nil function.
func (v *FuncValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *FuncValue) Get() uintptr { return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *FuncValue) Set(x *FuncValue) {
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *FuncValue) SetValue(x Value) { v.Set(x.(*FuncValue)) }

// Method returns a FuncValue corresponding to v's i'th method.
// The arguments to a Call on the returned FuncValue
// should not include a receiver; the FuncValue will use v
// as the receiver.
func (v *value) Method(i int) *FuncValue {
	t := v.Type().uncommon()
	if t == nil || i < 0 || i >= len(t.methods) {
		return nil
	}
	p := &t.methods[i]
	fn := p.tfn
	fv := &FuncValue{value: value{toType(*p.typ), addr(&fn), true}, first: v, isInterface: false}
	return fv
}

// implemented in ../pkg/runtime/*/asm.s
func call(fn, arg *byte, n uint32)

type tiny struct {
	b byte
}

// Call calls the function fv with input parameters in.
// It returns the function's output parameters as Values.
func (fv *FuncValue) Call(in []Value) []Value {
	var structAlign = Typeof((*tiny)(nil)).(*PtrType).Elem().Size()

	t := fv.Type().(*FuncType)
	nin := len(in)
	if fv.first != nil && !fv.isInterface {
		nin++
	}
	if nin != t.NumIn() {
		panic("FuncValue: wrong argument count")
	}
	nout := t.NumOut()

	// Compute arg size & allocate.
	// This computation is 6g/8g-dependent
	// and probably wrong for gccgo, but so
	// is most of this function.
	size := uintptr(0)
	if fv.isInterface {
		// extra word for interface value
		size += ptrSize
	}
	for i := 0; i < nin; i++ {
		tv := t.In(i)
		a := uintptr(tv.Align())
		size = (size + a - 1) &^ (a - 1)
		size += tv.Size()
	}
	size = (size + structAlign - 1) &^ (structAlign - 1)
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		a := uintptr(tv.Align())
		size = (size + a - 1) &^ (a - 1)
		size += tv.Size()
	}

	// size must be > 0 in order for &args[0] to be valid.
	// the argument copying is going to round it up to
	// a multiple of 8 anyway, so make it 8 to begin with.
	if size < 8 {
		size = 8
	}

	// round to pointer size
	size = (size + ptrSize - 1) &^ (ptrSize - 1)

	// Copy into args.
	//
	// TODO(rsc): revisit when reference counting happens.
	// The values are holding up the in references for us,
	// but something must be done for the out references.
	// For now make everything look like a pointer by pretending
	// to allocate a []*int.
	args := make([]*int, size/ptrSize)
	ptr := uintptr(unsafe.Pointer(&args[0]))
	off := uintptr(0)
	delta := 0
	if v := fv.first; v != nil {
		// Hard-wired first argument.
		if fv.isInterface {
			// v is a single uninterpreted word
			memmove(addr(ptr), v.getAddr(), ptrSize)
			off = ptrSize
		} else {
			// v is a real value
			tv := v.Type()
			typesMustMatch(t.In(0), tv)
			n := tv.Size()
			memmove(addr(ptr), v.getAddr(), n)
			off = n
			delta = 1
		}
	}
	for i, v := range in {
		tv := v.Type()
		typesMustMatch(t.In(i+delta), tv)
		a := uintptr(tv.Align())
		off = (off + a - 1) &^ (a - 1)
		n := tv.Size()
		memmove(addr(ptr+off), v.getAddr(), n)
		off += n
	}
	off = (off + structAlign - 1) &^ (structAlign - 1)

	// Call
	call(*(**byte)(fv.addr), (*byte)(addr(ptr)), uint32(size))

	// Copy return values out of args.
	//
	// TODO(rsc): revisit like above.
	ret := make([]Value, nout)
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		a := uintptr(tv.Align())
		off = (off + a - 1) &^ (a - 1)
		v := MakeZero(tv)
		n := tv.Size()
		memmove(v.getAddr(), addr(ptr+off), n)
		ret[i] = v
		off += n
	}

	return ret
}

/*
 * interface
 */

// An InterfaceValue represents an interface value.
type InterfaceValue struct {
	value "interface"
}

// No Get because v.Interface() is available.

// IsNil returns whether v is a nil interface value.
func (v *InterfaceValue) IsNil() bool { return v.Interface() == nil }

// Elem returns the concrete value stored in the interface value v.
func (v *InterfaceValue) Elem() Value { return NewValue(v.Interface()) }

// ../runtime/reflect.cgo
func setiface(typ *InterfaceType, x *interface{}, addr addr)

// Set assigns x to v.
func (v *InterfaceValue) Set(x Value) {
	var i interface{}
	if x != nil {
		i = x.Interface()
	}
	if !v.canSet {
		panic(cannotSet)
	}
	// Two different representations; see comment in Get.
	// Empty interface is easy.
	t := v.typ.(*InterfaceType)
	if t.NumMethod() == 0 {
		*(*interface{})(v.addr) = i
		return
	}

	// Non-empty interface requires a runtime check.
	setiface(t, &i, v.addr)
}

// Set sets v to the value x.
func (v *InterfaceValue) SetValue(x Value) { v.Set(x) }

// Method returns a FuncValue corresponding to v's i'th method.
// The arguments to a Call on the returned FuncValue
// should not include a receiver; the FuncValue will use v
// as the receiver.
func (v *InterfaceValue) Method(i int) *FuncValue {
	t := v.Type().(*InterfaceType)
	if t == nil || i < 0 || i >= len(t.methods) {
		return nil
	}
	p := &t.methods[i]

	// Interface is two words: itable, data.
	tab := *(**runtime.Itable)(v.addr)
	data := &value{Typeof((*byte)(nil)), addr(uintptr(v.addr) + ptrSize), true}

	// Function pointer is at p.perm in the table.
	fn := tab.Fn[i]
	fv := &FuncValue{value: value{toType(*p.typ), addr(&fn), true}, first: data, isInterface: true}
	return fv
}

/*
 * map
 */

// A MapValue represents a map value.
type MapValue struct {
	value "map"
}

// IsNil returns whether v is a nil map value.
func (v *MapValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *MapValue) Set(x *MapValue) {
	if !v.canSet {
		panic(cannotSet)
	}
	if x == nil {
		*(**uintptr)(v.addr) = nil
		return
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *MapValue) SetValue(x Value) {
	if x == nil {
		v.Set(nil)
		return
	}
	v.Set(x.(*MapValue))
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *MapValue) Get() uintptr { return *(*uintptr)(v.addr) }

// implemented in ../pkg/runtime/reflect.cgo
func mapaccess(m, key, val *byte) bool
func mapassign(m, key, val *byte)
func maplen(m *byte) int32
func mapiterinit(m *byte) *byte
func mapiternext(it *byte)
func mapiterkey(it *byte, key *byte) bool
func makemap(t *runtime.MapType) *byte

// Elem returns the value associated with key in the map v.
// It returns nil if key is not found in the map.
func (v *MapValue) Elem(key Value) Value {
	t := v.Type().(*MapType)
	typesMustMatch(t.Key(), key.Type())
	m := *(**byte)(v.addr)
	if m == nil {
		return nil
	}
	newval := MakeZero(t.Elem())
	if !mapaccess(m, (*byte)(key.getAddr()), (*byte)(newval.getAddr())) {
		return nil
	}
	return newval
}

// SetElem sets the value associated with key in the map v to val.
// If val is nil, Put deletes the key from map.
func (v *MapValue) SetElem(key, val Value) {
	t := v.Type().(*MapType)
	typesMustMatch(t.Key(), key.Type())
	var vaddr *byte
	if val != nil {
		typesMustMatch(t.Elem(), val.Type())
		vaddr = (*byte)(val.getAddr())
	}
	m := *(**byte)(v.addr)
	mapassign(m, (*byte)(key.getAddr()), vaddr)
}

// Len returns the number of keys in the map v.
func (v *MapValue) Len() int {
	m := *(**byte)(v.addr)
	if m == nil {
		return 0
	}
	return int(maplen(m))
}

// Keys returns a slice containing all the keys present in the map,
// in unspecified order.
func (v *MapValue) Keys() []Value {
	tk := v.Type().(*MapType).Key()
	m := *(**byte)(v.addr)
	mlen := int32(0)
	if m != nil {
		mlen = maplen(m)
	}
	it := mapiterinit(m)
	a := make([]Value, mlen)
	var i int
	for i = 0; i < len(a); i++ {
		k := MakeZero(tk)
		if !mapiterkey(it, (*byte)(k.getAddr())) {
			break
		}
		a[i] = k
		mapiternext(it)
	}
	return a[0:i]
}

// MakeMap creates a new map of the specified type.
func MakeMap(typ *MapType) *MapValue {
	v := MakeZero(typ).(*MapValue)
	*(**byte)(v.addr) = makemap((*runtime.MapType)(unsafe.Pointer(typ)))
	return v
}

/*
 * ptr
 */

// A PtrValue represents a pointer.
type PtrValue struct {
	value "ptr"
}

// IsNil returns whether v is a nil pointer.
func (v *PtrValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *PtrValue) Get() uintptr { return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *PtrValue) Set(x *PtrValue) {
	if x == nil {
		*(**uintptr)(v.addr) = nil
		return
	}
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	// TODO: This will have to move into the runtime
	// once the new gc goes in
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *PtrValue) SetValue(x Value) {
	if x == nil {
		v.Set(nil)
		return
	}
	v.Set(x.(*PtrValue))
}

// PointTo changes v to point to x.
// If x is a nil Value, PointTo sets v to nil.
func (v *PtrValue) PointTo(x Value) {
	if x == nil {
		*(**uintptr)(v.addr) = nil
		return
	}
	if !x.CanSet() {
		panic("cannot set x; cannot point to x")
	}
	typesMustMatch(v.typ.(*PtrType).Elem(), x.Type())
	// TODO: This will have to move into the runtime
	// once the new gc goes in.
	*(*uintptr)(v.addr) = x.Addr()
}

// Elem returns the value that v points to.
// If v is a nil pointer, Elem returns a nil Value.
func (v *PtrValue) Elem() Value {
	if v.IsNil() {
		return nil
	}
	return newValue(v.typ.(*PtrType).Elem(), *(*addr)(v.addr), v.canSet)
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a nil Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if pv, ok := v.(*PtrValue); ok {
		return pv.Elem()
	}
	return v
}

/*
 * struct
 */

// A StructValue represents a struct value.
type StructValue struct {
	value "struct"
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *StructValue) Set(x *StructValue) {
	// TODO: This will have to move into the runtime
	// once the gc goes in.
	if !v.canSet {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	memmove(v.addr, x.addr, v.typ.Size())
}

// Set sets v to the value x.
func (v *StructValue) SetValue(x Value) { v.Set(x.(*StructValue)) }

// Field returns the i'th field of the struct.
func (v *StructValue) Field(i int) Value {
	t := v.typ.(*StructType)
	if i < 0 || i >= t.NumField() {
		return nil
	}
	f := t.Field(i)
	return newValue(f.Type, addr(uintptr(v.addr)+f.Offset), v.canSet && f.PkgPath == "")
}

// FieldByIndex returns the nested field corresponding to index.
func (t *StructValue) FieldByIndex(index []int) (v Value) {
	v = t
	for i, x := range index {
		if i > 0 {
			if p, ok := v.(*PtrValue); ok {
				v = p.Elem()
			}
			if s, ok := v.(*StructValue); ok {
				t = s
			} else {
				v = nil
				return
			}
		}
		v = t.Field(x)
	}
	return
}

// FieldByName returns the struct field with the given name.
// The result is nil if no field was found.
func (t *StructValue) FieldByName(name string) Value {
	if f, ok := t.Type().(*StructType).FieldByName(name); ok {
		return t.FieldByIndex(f.Index)
	}
	return nil
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function.
// The result is nil if no field was found.
func (t *StructValue) FieldByNameFunc(match func(string) bool) Value {
	if f, ok := t.Type().(*StructType).FieldByNameFunc(match); ok {
		return t.FieldByIndex(f.Index)
	}
	return nil
}

// NumField returns the number of fields in the struct.
func (v *StructValue) NumField() int { return v.typ.(*StructType).NumField() }

/*
 * constructors
 */

// NewValue returns a new Value initialized to the concrete value
// stored in the interface i.  NewValue(nil) returns nil.
func NewValue(i interface{}) Value {
	if i == nil {
		return nil
	}
	t, a := unsafe.Reflect(i)
	return newValue(toType(t), addr(a), true)
}

func newValue(typ Type, addr addr, canSet bool) Value {
	v := value{typ, addr, canSet}
	switch typ.(type) {
	case *ArrayType:
		return &ArrayValue{v}
	case *BoolType:
		return &BoolValue{v}
	case *ChanType:
		return &ChanValue{v}
	case *FloatType:
		return &FloatValue{v}
	case *FuncType:
		return &FuncValue{value: v}
	case *ComplexType:
		return &ComplexValue{v}
	case *IntType:
		return &IntValue{v}
	case *InterfaceType:
		return &InterfaceValue{v}
	case *MapType:
		return &MapValue{v}
	case *PtrType:
		return &PtrValue{v}
	case *SliceType:
		return &SliceValue{v}
	case *StringType:
		return &StringValue{v}
	case *StructType:
		return &StructValue{v}
	case *UintType:
		return &UintValue{v}
	case *UnsafePointerType:
		return &UnsafePointerValue{v}
	}
	panic("newValue" + typ.String())
}

// MakeZero returns a zero Value for the specified Type.
func MakeZero(typ Type) Value {
	if typ == nil {
		return nil
	}
	return newValue(typ, addr(unsafe.New(typ)), true)
}
