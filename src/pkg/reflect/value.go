// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"runtime";
	"unsafe";
)

const ptrSize = uintptr(unsafe.Sizeof((*byte)(nil)))
const cannotSet = "cannot set value obtained via unexported struct field"

type addr unsafe.Pointer

// TODO: This will have to go away when
// the new gc goes in.
func memmove(adst, asrc addr, n uintptr) {
	dst := uintptr(adst);
	src := uintptr(asrc);
	switch {
	case src < dst && src+n > dst:
		// byte copy backward
		// careful: i is unsigned
		for i := n; i > 0; {
			i--;
			*(*byte)(addr(dst+i)) = *(*byte)(addr(src+i));
		}
	case (n|src|dst)&(ptrSize-1) != 0:
		// byte copy forward
		for i := uintptr(0); i < n; i++ {
			*(*byte)(addr(dst+i)) = *(*byte)(addr(src+i));
		}
	default:
		// word copy forward
		for i := uintptr(0); i < n; i += ptrSize {
			*(*uintptr)(addr(dst+i)) = *(*uintptr)(addr(src+i));
		}
	}
}

// Value is the common interface to reflection values.
// The implementations of Value (e.g., ArrayValue, StructValue)
// have additional type-specific methods.
type Value interface {
	// Type returns the value's type.
	Type() Type;

	// Interface returns the value as an interface{}.
	Interface() interface{};

	// CanSet returns whether the value can be changed.
	// Values obtained by the use of non-exported struct fields
	// can be used in Get but not Set.
	// If CanSet() returns false, calling the type-specific Set
	// will cause a crash.
	CanSet() bool;

	// SetValue assigns v to the value; v must have the same type as the value.
	SetValue(v Value);

	// Addr returns a pointer to the underlying data.
	// It is for advanced clients that also
	// import the "unsafe" package.
	Addr() uintptr;

	// Method returns a FuncValue corresponding to the value's i'th method.
	// The arguments to a Call on the returned FuncValue
	// should not include a receiver; the FuncValue will use
	// the value as the receiver.
	Method(i int) *FuncValue;

	getAddr() addr;
}

type value struct {
	typ	Type;
	addr	addr;
	canSet	bool;
}

func (v *value) Type() Type	{ return v.typ }

func (v *value) Addr() uintptr	{ return uintptr(v.addr) }

func (v *value) getAddr() addr	{ return v.addr }

func (v *value) Interface() interface{} {
	if typ, ok := v.typ.(*InterfaceType); ok {
		// There are two different representations of interface values,
		// one if the interface type has methods and one if it doesn't.
		// These two representations require different expressions
		// to extract correctly.
		if typ.NumMethod() == 0 {
			// Extract as interface value without methods.
			return *(*interface{})(v.addr);
		}
		// Extract from v.addr as interface value with methods.
		return *(*interface {
			m();
		})(v.addr);
	}
	return unsafe.Unreflect(v.typ, unsafe.Pointer(v.addr));
}

func (v *value) CanSet() bool	{ return v.canSet }

/*
 * basic types
 */

// BoolValue represents a bool value.
type BoolValue struct {
	value;
}

// Get returns the underlying bool value.
func (v *BoolValue) Get() bool	{ return *(*bool)(v.addr) }

// Set sets v to the value x.
func (v *BoolValue) Set(x bool) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*bool)(v.addr) = x;
}

// Set sets v to the value x.
func (v *BoolValue) SetValue(x Value)	{ v.Set(x.(*BoolValue).Get()) }

// FloatValue represents a float value.
type FloatValue struct {
	value;
}

// Get returns the underlying float value.
func (v *FloatValue) Get() float	{ return *(*float)(v.addr) }

// Set sets v to the value x.
func (v *FloatValue) Set(x float) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float)(v.addr) = x;
}

// Set sets v to the value x.
func (v *FloatValue) SetValue(x Value)	{ v.Set(x.(*FloatValue).Get()) }

// Float32Value represents a float32 value.
type Float32Value struct {
	value;
}

// Get returns the underlying float32 value.
func (v *Float32Value) Get() float32	{ return *(*float32)(v.addr) }

// Set sets v to the value x.
func (v *Float32Value) Set(x float32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float32)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Float32Value) SetValue(x Value)	{ v.Set(x.(*Float32Value).Get()) }

// Float64Value represents a float64 value.
type Float64Value struct {
	value;
}

// Get returns the underlying float64 value.
func (v *Float64Value) Get() float64	{ return *(*float64)(v.addr) }

// Set sets v to the value x.
func (v *Float64Value) Set(x float64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float64)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Float64Value) SetValue(x Value)	{ v.Set(x.(*Float64Value).Get()) }

// IntValue represents an int value.
type IntValue struct {
	value;
}

// Get returns the underlying int value.
func (v *IntValue) Get() int	{ return *(*int)(v.addr) }

// Set sets v to the value x.
func (v *IntValue) Set(x int) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int)(v.addr) = x;
}

// Set sets v to the value x.
func (v *IntValue) SetValue(x Value)	{ v.Set(x.(*IntValue).Get()) }

// Int8Value represents an int8 value.
type Int8Value struct {
	value;
}

// Get returns the underlying int8 value.
func (v *Int8Value) Get() int8	{ return *(*int8)(v.addr) }

// Set sets v to the value x.
func (v *Int8Value) Set(x int8) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int8)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Int8Value) SetValue(x Value)	{ v.Set(x.(*Int8Value).Get()) }

// Int16Value represents an int16 value.
type Int16Value struct {
	value;
}

// Get returns the underlying int16 value.
func (v *Int16Value) Get() int16	{ return *(*int16)(v.addr) }

// Set sets v to the value x.
func (v *Int16Value) Set(x int16) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int16)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Int16Value) SetValue(x Value)	{ v.Set(x.(*Int16Value).Get()) }

// Int32Value represents an int32 value.
type Int32Value struct {
	value;
}

// Get returns the underlying int32 value.
func (v *Int32Value) Get() int32	{ return *(*int32)(v.addr) }

// Set sets v to the value x.
func (v *Int32Value) Set(x int32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int32)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Int32Value) SetValue(x Value)	{ v.Set(x.(*Int32Value).Get()) }

// Int64Value represents an int64 value.
type Int64Value struct {
	value;
}

// Get returns the underlying int64 value.
func (v *Int64Value) Get() int64	{ return *(*int64)(v.addr) }

// Set sets v to the value x.
func (v *Int64Value) Set(x int64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int64)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Int64Value) SetValue(x Value)	{ v.Set(x.(*Int64Value).Get()) }

// StringValue represents a string value.
type StringValue struct {
	value;
}

// Get returns the underlying string value.
func (v *StringValue) Get() string	{ return *(*string)(v.addr) }

// Set sets v to the value x.
func (v *StringValue) Set(x string) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*string)(v.addr) = x;
}

// Set sets v to the value x.
func (v *StringValue) SetValue(x Value)	{ v.Set(x.(*StringValue).Get()) }

// UintValue represents a uint value.
type UintValue struct {
	value;
}

// Get returns the underlying uint value.
func (v *UintValue) Get() uint	{ return *(*uint)(v.addr) }

// Set sets v to the value x.
func (v *UintValue) Set(x uint) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint)(v.addr) = x;
}

// Set sets v to the value x.
func (v *UintValue) SetValue(x Value)	{ v.Set(x.(*UintValue).Get()) }

// Uint8Value represents a uint8 value.
type Uint8Value struct {
	value;
}

// Get returns the underlying uint8 value.
func (v *Uint8Value) Get() uint8	{ return *(*uint8)(v.addr) }

// Set sets v to the value x.
func (v *Uint8Value) Set(x uint8) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint8)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Uint8Value) SetValue(x Value)	{ v.Set(x.(*Uint8Value).Get()) }

// Uint16Value represents a uint16 value.
type Uint16Value struct {
	value;
}

// Get returns the underlying uint16 value.
func (v *Uint16Value) Get() uint16	{ return *(*uint16)(v.addr) }

// Set sets v to the value x.
func (v *Uint16Value) Set(x uint16) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint16)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Uint16Value) SetValue(x Value)	{ v.Set(x.(*Uint16Value).Get()) }

// Uint32Value represents a uint32 value.
type Uint32Value struct {
	value;
}

// Get returns the underlying uint32 value.
func (v *Uint32Value) Get() uint32	{ return *(*uint32)(v.addr) }

// Set sets v to the value x.
func (v *Uint32Value) Set(x uint32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint32)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Uint32Value) SetValue(x Value)	{ v.Set(x.(*Uint32Value).Get()) }

// Uint64Value represents a uint64 value.
type Uint64Value struct {
	value;
}

// Get returns the underlying uint64 value.
func (v *Uint64Value) Get() uint64	{ return *(*uint64)(v.addr) }

// Set sets v to the value x.
func (v *Uint64Value) Set(x uint64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint64)(v.addr) = x;
}

// Set sets v to the value x.
func (v *Uint64Value) SetValue(x Value)	{ v.Set(x.(*Uint64Value).Get()) }

// UintptrValue represents a uintptr value.
type UintptrValue struct {
	value;
}

// Get returns the underlying uintptr value.
func (v *UintptrValue) Get() uintptr	{ return *(*uintptr)(v.addr) }

// Set sets v to the value x.
func (v *UintptrValue) Set(x uintptr) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uintptr)(v.addr) = x;
}

// Set sets v to the value x.
func (v *UintptrValue) SetValue(x Value)	{ v.Set(x.(*UintptrValue).Get()) }

// UnsafePointerValue represents an unsafe.Pointer value.
type UnsafePointerValue struct {
	value;
}

// Get returns the underlying uintptr value.
// Get returns uintptr, not unsafe.Pointer, so that
// programs that do not import "unsafe" cannot
// obtain a value of unsafe.Pointer type from "reflect".
func (v *UnsafePointerValue) Get() uintptr	{ return uintptr(*(*unsafe.Pointer)(v.addr)) }

// Set sets v to the value x.
func (v *UnsafePointerValue) Set(x unsafe.Pointer) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*unsafe.Pointer)(v.addr) = x;
}

// Set sets v to the value x.
func (v *UnsafePointerValue) SetValue(x Value) {
	v.Set(unsafe.Pointer(x.(*UnsafePointerValue).Get()));
}

func typesMustMatch(t1, t2 Type) {
	if t1 != t2 {
		panicln("type mismatch:", t1.String(), "!=", t2.String());
	}
}

/*
 * array
 */

// ArrayOrSliceValue is the common interface
// implemented by both ArrayValue and SliceValue.
type ArrayOrSliceValue interface {
	Value;
	Len() int;
	Cap() int;
	Elem(i int) Value;
	addr() addr;
}

// ArrayCopy copies the contents of src into dst until either
// dst has been filled or src has been exhausted.
// It returns the number of elements copied.
// The arrays dst and src must have the same element type.
func ArrayCopy(dst, src ArrayOrSliceValue) int {
	// TODO: This will have to move into the runtime
	// once the real gc goes in.
	de := dst.Type().(ArrayOrSliceType).Elem();
	se := src.Type().(ArrayOrSliceType).Elem();
	typesMustMatch(de, se);
	n := dst.Len();
	if xn := src.Len(); n > xn {
		n = xn;
	}
	memmove(dst.addr(), src.addr(), uintptr(n) * de.Size());
	return n;
}

// An ArrayValue represents an array.
type ArrayValue struct {
	value;
}

// Len returns the length of the array.
func (v *ArrayValue) Len() int	{ return v.typ.(*ArrayType).Len() }

// Cap returns the capacity of the array (equal to Len()).
func (v *ArrayValue) Cap() int	{ return v.typ.(*ArrayType).Len() }

// addr returns the base address of the data in the array.
func (v *ArrayValue) addr() addr	{ return v.value.addr }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ArrayValue) Set(x *ArrayValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	ArrayCopy(v, x);
}

// Set sets v to the value x.
func (v *ArrayValue) SetValue(x Value)	{ v.Set(x.(*ArrayValue)) }

// Elem returns the i'th element of v.
func (v *ArrayValue) Elem(i int) Value {
	typ := v.typ.(*ArrayType).Elem();
	n := v.Len();
	if i < 0 || i >= n {
		panic("index", i, "in array len", n);
	}
	p := addr(uintptr(v.addr()) + uintptr(i) * typ.Size());
	return newValue(typ, p, v.canSet);
}

/*
 * slice
 */

// runtime representation of slice
type SliceHeader struct {
	Data	uintptr;
	Len	int;
	Cap	int;
}

// A SliceValue represents a slice.
type SliceValue struct {
	value;
}

func (v *SliceValue) slice() *SliceHeader	{ return (*SliceHeader)(v.value.addr) }

// IsNil returns whether v is a nil slice.
func (v *SliceValue) IsNil() bool	{ return v.slice().Data == 0 }

// Len returns the length of the slice.
func (v *SliceValue) Len() int	{ return int(v.slice().Len) }

// Cap returns the capacity of the slice.
func (v *SliceValue) Cap() int	{ return int(v.slice().Cap) }

// addr returns the base address of the data in the slice.
func (v *SliceValue) addr() addr	{ return addr(v.slice().Data) }

// SetLen changes the length of v.
// The new length n must be between 0 and the capacity, inclusive.
func (v *SliceValue) SetLen(n int) {
	s := v.slice();
	if n < 0 || n > int(s.Cap) {
		panicln("SetLen", n, "with capacity", s.Cap);
	}
	s.Len = n;
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *SliceValue) Set(x *SliceValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*v.slice() = *x.slice();
}

// Set sets v to the value x.
func (v *SliceValue) SetValue(x Value)	{ v.Set(x.(*SliceValue)) }

// Slice returns a sub-slice of the slice v.
func (v *SliceValue) Slice(beg, end int) *SliceValue {
	cap := v.Cap();
	if beg < 0 || end < beg || end > cap {
		panic("slice bounds [", beg, ":", end, "] with capacity ", cap);
	}
	typ := v.typ.(*SliceType);
	s := new(SliceHeader);
	s.Data = uintptr(v.addr()) + uintptr(beg) * typ.Elem().Size();
	s.Len = end-beg;
	s.Cap = cap-beg;
	return newValue(typ, addr(s), v.canSet).(*SliceValue);
}

// Elem returns the i'th element of v.
func (v *SliceValue) Elem(i int) Value {
	typ := v.typ.(*SliceType).Elem();
	n := v.Len();
	if i < 0 || i >= n {
		panicln("index", i, "in array of length", n);
	}
	p := addr(uintptr(v.addr()) + uintptr(i) * typ.Size());
	return newValue(typ, p, v.canSet);
}

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ *SliceType, len, cap int) *SliceValue {
	s := new(SliceHeader);
	size := typ.Elem().Size() * uintptr(cap);
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	s.Data = uintptr(addr(&data[0]));
	s.Len = len;
	s.Cap = cap;
	return newValue(typ, addr(s), true).(*SliceValue);
}

/*
 * chan
 */

// A ChanValue represents a chan.
type ChanValue struct {
	value;
}

// IsNil returns whether v is a nil channel.
func (v *ChanValue) IsNil() bool	{ return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ChanValue) Set(x *ChanValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Set sets v to the value x.
func (v *ChanValue) SetValue(x Value)	{ v.Set(x.(*ChanValue)) }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *ChanValue) Get() uintptr	{ return *(*uintptr)(v.addr) }

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
	ch := *(**byte)(v.addr);
	return chanclosed(ch);
}

// Close closes the channel.
func (v *ChanValue) Close() {
	ch := *(**byte)(v.addr);
	chanclose(ch);
}

func (v *ChanValue) Len() int {
	ch := *(**byte)(v.addr);
	return int(chanlen(ch));
}

func (v *ChanValue) Cap() int {
	ch := *(**byte)(v.addr);
	return int(chancap(ch));
}

// internal send; non-blocking if b != nil
func (v *ChanValue) send(x Value, b *bool) {
	t := v.Type().(*ChanType);
	if t.Dir() & SendDir == 0 {
		panic("send on recv-only channel");
	}
	typesMustMatch(t.Elem(), x.Type());
	ch := *(**byte)(v.addr);
	chansend(ch, (*byte)(x.getAddr()), b);
}

// internal recv; non-blocking if b != nil
func (v *ChanValue) recv(b *bool) Value {
	t := v.Type().(*ChanType);
	if t.Dir() & RecvDir == 0 {
		panic("recv on send-only channel");
	}
	ch := *(**byte)(v.addr);
	x := MakeZero(t.Elem());
	chanrecv(ch, (*byte)(x.getAddr()), b);
	return x;
}

// Send sends x on the channel v.
func (v *ChanValue) Send(x Value)	{ v.send(x, nil) }

// Recv receives and returns a value from the channel v.
func (v *ChanValue) Recv() Value	{ return v.recv(nil) }

// TrySend attempts to sends x on the channel v but will not block.
// It returns true if the value was sent, false otherwise.
func (v *ChanValue) TrySend(x Value) bool {
	var ok bool;
	v.send(x, &ok);
	return ok;
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It returns the value if one is received, nil otherwise.
func (v *ChanValue) TryRecv() Value {
	var ok bool;
	x := v.recv(&ok);
	if !ok {
		return nil;
	}
	return x;
}

// MakeChan creates a new channel with the specified type and buffer size.
func MakeChan(typ *ChanType, buffer int) *ChanValue {
	if buffer < 0 {
		panic("MakeChan: negative buffer size");
	}
	if typ.Dir() != BothDir {
		panic("MakeChan: unidirectional channel type");
	}
	v := MakeZero(typ).(*ChanValue);
	*(**byte)(v.addr) = makechan((*runtime.ChanType)(unsafe.Pointer(typ)), uint32(buffer));
	return v;
}

/*
 * func
 */

// A FuncValue represents a function value.
type FuncValue struct {
	value;
	first		*value;
	isInterface	bool;
}

// IsNil returns whether v is a nil function.
func (v *FuncValue) IsNil() bool	{ return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *FuncValue) Get() uintptr	{ return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *FuncValue) Set(x *FuncValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Set sets v to the value x.
func (v *FuncValue) SetValue(x Value)	{ v.Set(x.(*FuncValue)) }

// Method returns a FuncValue corresponding to v's i'th method.
// The arguments to a Call on the returned FuncValue
// should not include a receiver; the FuncValue will use v
// as the receiver.
func (v *value) Method(i int) *FuncValue {
	t := v.Type().uncommon();
	if t == nil || i < 0 || i >= len(t.methods) {
		return nil;
	}
	p := &t.methods[i];
	fn := p.tfn;
	fv := &FuncValue{value: value{toType(*p.typ), addr(&fn), true}, first: v, isInterface: false};
	return fv;
}

// implemented in ../pkg/runtime/*/asm.s
func call(fn, arg *byte, n uint32)

type tiny struct {
	b byte;
}

// Call calls the function v with input parameters in.
// It returns the function's output parameters as Values.
func (fv *FuncValue) Call(in []Value) []Value {
	var structAlign = Typeof((*tiny)(nil)).(*PtrType).Elem().Size();

	t := fv.Type().(*FuncType);
	nin := len(in);
	if fv.first != nil && !fv.isInterface {
		nin++;
	}
	if nin != t.NumIn() {
		panic("FuncValue: wrong argument count");
	}
	nout := t.NumOut();

	// Compute arg size & allocate.
	// This computation is 6g/8g-dependent
	// and probably wrong for gccgo, but so
	// is most of this function.
	size := uintptr(0);
	if fv.isInterface {
		// extra word for interface value
		size += ptrSize;
	}
	for i := 0; i < nin; i++ {
		tv := t.In(i);
		a := uintptr(tv.Align());
		size = (size+a-1)&^(a-1);
		size += tv.Size();
	}
	size = (size + structAlign - 1)&^(structAlign - 1);
	for i := 0; i < nout; i++ {
		tv := t.Out(i);
		a := uintptr(tv.Align());
		size = (size+a-1)&^(a-1);
		size += tv.Size();
	}

	// size must be > 0 in order for &args[0] to be valid.
	// the argument copying is going to round it up to
	// a multiple of 8 anyway, so make it 8 to begin with.
	if size < 8 {
		size = 8;
	}
	args := make([]byte, size);
	ptr := uintptr(unsafe.Pointer(&args[0]));

	// Copy into args.
	//
	// TODO(rsc): revisit when reference counting happens.
	// This one may be fine.  The values are holding up the
	// references for us, so maybe this can be treated
	// like any stack-to-stack copy.
	off := uintptr(0);
	delta := 0;
	if v := fv.first; v != nil {
		// Hard-wired first argument.
		if fv.isInterface {
			// v is a single uninterpreted word
			memmove(addr(ptr), v.getAddr(), ptrSize);
			off = ptrSize;
		} else {
			// v is a real value
			tv := v.Type();
			typesMustMatch(t.In(0), tv);
			n := tv.Size();
			memmove(addr(ptr), v.getAddr(), n);
			off = n;
			delta = 1;
		}
	}
	for i, v := range in {
		tv := v.Type();
		typesMustMatch(t.In(i+delta), tv);
		a := uintptr(tv.Align());
		off = (off+a-1)&^(a-1);
		n := tv.Size();
		memmove(addr(ptr+off), v.getAddr(), n);
		off += n;
	}
	off = (off + structAlign - 1)&^(structAlign - 1);

	// Call
	call(*(**byte)(fv.addr), (*byte)(addr(ptr)), uint32(size));

	// Copy return values out of args.
	//
	// TODO(rsc): revisit like above.
	ret := make([]Value, nout);
	for i := 0; i < nout; i++ {
		tv := t.Out(i);
		a := uintptr(tv.Align());
		off = (off+a-1)&^(a-1);
		v := MakeZero(tv);
		n := tv.Size();
		memmove(v.getAddr(), addr(ptr+off), n);
		ret[i] = v;
		off += n;
	}

	return ret;
}

/*
 * interface
 */

// An InterfaceValue represents an interface value.
type InterfaceValue struct {
	value;
}

// No Get because v.Interface() is available.

// IsNil returns whether v is a nil interface value.
func (v *InterfaceValue) IsNil() bool	{ return v.Interface() == nil }

// Elem returns the concrete value stored in the interface value v.
func (v *InterfaceValue) Elem() Value	{ return NewValue(v.Interface()) }

// ../runtime/reflect.cgo
func setiface(typ *InterfaceType, x *interface{}, addr addr)

// Set assigns x to v.
func (v *InterfaceValue) Set(x Value) {
	i := x.Interface();
	if !v.canSet {
		panic(cannotSet);
	}
	// Two different representations; see comment in Get.
	// Empty interface is easy.
	t := v.typ.(*InterfaceType);
	if t.NumMethod() == 0 {
		*(*interface{})(v.addr) = i;
		return;
	}

	// Non-empty interface requires a runtime check.
	setiface(t, &i, v.addr);
}

// Set sets v to the value x.
func (v *InterfaceValue) SetValue(x Value)	{ v.Set(x) }

// Method returns a FuncValue corresponding to v's i'th method.
// The arguments to a Call on the returned FuncValue
// should not include a receiver; the FuncValue will use v
// as the receiver.
func (v *InterfaceValue) Method(i int) *FuncValue {
	t := v.Type().(*InterfaceType);
	if t == nil || i < 0 || i >= len(t.methods) {
		return nil;
	}
	p := &t.methods[i];

	// Interface is two words: itable, data.
	tab := *(**runtime.Itable)(v.addr);
	data := &value{Typeof((*byte)(nil)), addr(uintptr(v.addr)+ptrSize), true};

	// Function pointer is at p.perm in the table.
	fn := tab.Fn[p.perm];
	fv := &FuncValue{value: value{toType(*p.typ), addr(&fn), true}, first: data, isInterface: true};
	return fv;
}

/*
 * map
 */

// A MapValue represents a map value.
type MapValue struct {
	value;
}

// IsNil returns whether v is a nil map value.
func (v *MapValue) IsNil() bool	{ return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *MapValue) Set(x *MapValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Set sets v to the value x.
func (v *MapValue) SetValue(x Value)	{ v.Set(x.(*MapValue)) }

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
	t := v.Type().(*MapType);
	typesMustMatch(t.Key(), key.Type());
	m := *(**byte)(v.addr);
	if m == nil {
		return nil;
	}
	newval := MakeZero(t.Elem());
	if !mapaccess(m, (*byte)(key.getAddr()), (*byte)(newval.getAddr())) {
		return nil;
	}
	return newval;
}

// SetElem sets the value associated with key in the map v to val.
// If val is nil, Put deletes the key from map.
func (v *MapValue) SetElem(key, val Value) {
	t := v.Type().(*MapType);
	typesMustMatch(t.Key(), key.Type());
	var vaddr *byte;
	if val != nil {
		typesMustMatch(t.Elem(), val.Type());
		vaddr = (*byte)(val.getAddr());
	}
	m := *(**byte)(v.addr);
	mapassign(m, (*byte)(key.getAddr()), vaddr);
}

// Len returns the number of keys in the map v.
func (v *MapValue) Len() int {
	m := *(**byte)(v.addr);
	if m == nil {
		return 0;
	}
	return int(maplen(m));
}

// Keys returns a slice containing all the keys present in the map,
// in unspecified order.
func (v *MapValue) Keys() []Value {
	tk := v.Type().(*MapType).Key();
	m := *(**byte)(v.addr);
	mlen := int32(0);
	if m != nil {
		mlen = maplen(m);
	}
	it := mapiterinit(m);
	a := make([]Value, mlen);
	var i int;
	for i = 0; i < len(a); i++ {
		k := MakeZero(tk);
		if !mapiterkey(it, (*byte)(k.getAddr())) {
			break;
		}
		a[i] = k;
		mapiternext(it);
	}
	return a[0:i];
}

// MakeMap creates a new map of the specified type.
func MakeMap(typ *MapType) *MapValue {
	v := MakeZero(typ).(*MapValue);
	*(**byte)(v.addr) = makemap((*runtime.MapType)(unsafe.Pointer(typ)));
	return v;
}

/*
 * ptr
 */

// A PtrValue represents a pointer.
type PtrValue struct {
	value;
}

// IsNil returns whether v is a nil pointer.
func (v *PtrValue) IsNil() bool	{ return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *PtrValue) Get() uintptr	{ return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *PtrValue) Set(x *PtrValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	// TODO: This will have to move into the runtime
	// once the new gc goes in
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Set sets v to the value x.
func (v *PtrValue) SetValue(x Value)	{ v.Set(x.(*PtrValue)) }

// PointTo changes v to point to x.
func (v *PtrValue) PointTo(x Value) {
	if !x.CanSet() {
		panic("cannot set x; cannot point to x");
	}
	typesMustMatch(v.typ.(*PtrType).Elem(), x.Type());
	// TODO: This will have to move into the runtime
	// once the new gc goes in.
	*(*uintptr)(v.addr) = x.Addr();
}

// Elem returns the value that v points to.
// If v is a nil pointer, Elem returns a nil Value.
func (v *PtrValue) Elem() Value {
	if v.IsNil() {
		return nil;
	}
	return newValue(v.typ.(*PtrType).Elem(), *(*addr)(v.addr), v.canSet);
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a nil Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if pv, ok := v.(*PtrValue); ok {
		return pv.Elem();
	}
	return v;
}

/*
 * struct
 */

// A StructValue represents a struct value.
type StructValue struct {
	value;
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *StructValue) Set(x *StructValue) {
	// TODO: This will have to move into the runtime
	// once the gc goes in.
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	memmove(v.addr, x.addr, v.typ.Size());
}

// Set sets v to the value x.
func (v *StructValue) SetValue(x Value)	{ v.Set(x.(*StructValue)) }

// Field returns the i'th field of the struct.
func (v *StructValue) Field(i int) Value {
	t := v.typ.(*StructType);
	if i < 0 || i >= t.NumField() {
		return nil;
	}
	f := t.Field(i);
	return newValue(f.Type, addr(uintptr(v.addr) + f.Offset), v.canSet && f.PkgPath == "");
}

// FieldByIndex returns the nested field corresponding to index.
func (t *StructValue) FieldByIndex(index []int) (v Value) {
	v = t;
	for i, x := range index {
		if i > 0 {
			if p, ok := v.(*PtrValue); ok {
				v = p.Elem();
			}
			if s, ok := v.(*StructValue); ok {
				t = s;
			} else {
				v = nil;
				return;
			}
		}
		v = t.Field(x);
	}
	return;
}

// FieldByName returns the struct field with the given name.
// The result is nil if no field was found.
func (t *StructValue) FieldByName(name string) Value {
	if f, ok := t.Type().(*StructType).FieldByName(name); ok {
		return t.FieldByIndex(f.Index);
	}
	return nil;
}

// NumField returns the number of fields in the struct.
func (v *StructValue) NumField() int	{ return v.typ.(*StructType).NumField() }

/*
 * constructors
 */

// NewValue returns a new Value initialized to the concrete value
// stored in the interface i.  NewValue(nil) returns nil.
func NewValue(i interface{}) Value {
	if i == nil {
		return nil;
	}
	t, a := unsafe.Reflect(i);
	return newValue(toType(t), addr(a), true);
}


func newFuncValue(typ Type, addr addr, canSet bool) *FuncValue {
	return &FuncValue{value: value{typ, addr, canSet}};
}

func newValue(typ Type, addr addr, canSet bool) Value {
	// FuncValue has a different layout;
	// it needs a extra space for the fixed receivers.
	if _, ok := typ.(*FuncType); ok {
		return newFuncValue(typ, addr, canSet);
	}

	// All values have same memory layout;
	// build once and convert.
	v := &struct{ value }{value{typ, addr, canSet}};
	switch typ.(type) {
	case *ArrayType:
		// TODO(rsc): Something must prevent
		// clients of the package from doing
		// this same kind of cast.
		// We should be allowed because
		// they're our types.
		// Something about implicit assignment
		// to struct fields.
		return (*ArrayValue)(v);
	case *BoolType:
		return (*BoolValue)(v);
	case *ChanType:
		return (*ChanValue)(v);
	case *FloatType:
		return (*FloatValue)(v);
	case *Float32Type:
		return (*Float32Value)(v);
	case *Float64Type:
		return (*Float64Value)(v);
	case *IntType:
		return (*IntValue)(v);
	case *Int8Type:
		return (*Int8Value)(v);
	case *Int16Type:
		return (*Int16Value)(v);
	case *Int32Type:
		return (*Int32Value)(v);
	case *Int64Type:
		return (*Int64Value)(v);
	case *InterfaceType:
		return (*InterfaceValue)(v);
	case *MapType:
		return (*MapValue)(v);
	case *PtrType:
		return (*PtrValue)(v);
	case *SliceType:
		return (*SliceValue)(v);
	case *StringType:
		return (*StringValue)(v);
	case *StructType:
		return (*StructValue)(v);
	case *UintType:
		return (*UintValue)(v);
	case *Uint8Type:
		return (*Uint8Value)(v);
	case *Uint16Type:
		return (*Uint16Value)(v);
	case *Uint32Type:
		return (*Uint32Value)(v);
	case *Uint64Type:
		return (*Uint64Value)(v);
	case *UintptrType:
		return (*UintptrValue)(v);
	case *UnsafePointerType:
		return (*UnsafePointerValue)(v);
	}
	panicln("newValue", typ.String());
}

// MakeZero returns a zero Value for the specified Type.
func MakeZero(typ Type) Value {
	// TODO: this will have to move into
	// the runtime proper in order to play nicely
	// with the garbage collector.
	size := typ.Size();
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	return newValue(typ, addr(&data[0]), true);
}
