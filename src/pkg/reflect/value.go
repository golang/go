// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"math"
	"runtime"
	"strconv"
	"unsafe"
)

const bigEndian = false // can be smarter if we find a big-endian machine
const ptrSize = unsafe.Sizeof((*byte)(nil))
const cannotSet = "cannot set value obtained from unexported struct field"

// TODO: This will have to go away when
// the new gc goes in.
func memmove(adst, asrc unsafe.Pointer, n uintptr) {
	dst := uintptr(adst)
	src := uintptr(asrc)
	switch {
	case src < dst && src+n > dst:
		// byte copy backward
		// careful: i is unsigned
		for i := n; i > 0; {
			i--
			*(*byte)(unsafe.Pointer(dst + i)) = *(*byte)(unsafe.Pointer(src + i))
		}
	case (n|src|dst)&(ptrSize-1) != 0:
		// byte copy forward
		for i := uintptr(0); i < n; i++ {
			*(*byte)(unsafe.Pointer(dst + i)) = *(*byte)(unsafe.Pointer(src + i))
		}
	default:
		// word copy forward
		for i := uintptr(0); i < n; i += ptrSize {
			*(*uintptr)(unsafe.Pointer(dst + i)) = *(*uintptr)(unsafe.Pointer(src + i))
		}
	}
}

// Value is the reflection interface to a Go value.
//
// Not all methods apply to all kinds of values.  Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of value before
// calling kind-specific methods.  Calling a method
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
type Value struct {
	// typ holds the type of the value represented by a Value.
	typ *rtype

	// val holds the 1-word representation of the value.
	// If flag's flagIndir bit is set, then val is a pointer to the data.
	// Otherwise val is a word holding the actual data.
	// When the data is smaller than a word, it begins at
	// the first byte (in the memory address sense) of val.
	// We use unsafe.Pointer so that the garbage collector
	// knows that val could be a pointer.
	val unsafe.Pointer

	// flag holds metadata about the value.
	// The lowest bits are flag bits:
	//	- flagRO: obtained via unexported field, so read-only
	//	- flagIndir: val holds a pointer to the data
	//	- flagAddr: v.CanAddr is true (implies flagIndir)
	//	- flagMethod: v is a method value.
	// The next five bits give the Kind of the value.
	// This repeats typ.Kind() except for method values.
	// The remaining 23+ bits give a method number for method values.
	// If flag.kind() != Func, code can assume that flagMethod is unset.
	// If typ.size > ptrSize, code can assume that flagIndir is set.
	flag

	// A method value represents a curried method invocation
	// like r.Read for some receiver r.  The typ+val+flag bits describe
	// the receiver r, but the flag's Kind bits say Func (methods are
	// functions), and the top bits of the flag give the method number
	// in r's type's method table.
}

type flag uintptr

const (
	flagRO flag = 1 << iota
	flagIndir
	flagAddr
	flagMethod
	flagKindShift        = iota
	flagKindWidth        = 5 // there are 27 kinds
	flagKindMask    flag = 1<<flagKindWidth - 1
	flagMethodShift      = flagKindShift + flagKindWidth
)

func (f flag) kind() Kind {
	return Kind((f >> flagKindShift) & flagKindMask)
}

// A ValueError occurs when a Value method is invoked on
// a Value that does not support it.  Such cases are documented
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

// An iword is the word that would be stored in an
// interface to represent a given value v.  Specifically, if v is
// bigger than a pointer, its word is a pointer to v's data.
// Otherwise, its word holds the data stored
// in its leading bytes (so is not a pointer).
// Because the value sometimes holds a pointer, we use
// unsafe.Pointer to represent it, so that if iword appears
// in a struct, the garbage collector knows that might be
// a pointer.
type iword unsafe.Pointer

func (v Value) iword() iword {
	if v.flag&flagIndir != 0 && v.typ.size <= ptrSize {
		// Have indirect but want direct word.
		return loadIword(v.val, v.typ.size)
	}
	return iword(v.val)
}

// loadIword loads n bytes at p from memory into an iword.
func loadIword(p unsafe.Pointer, n uintptr) iword {
	// Run the copy ourselves instead of calling memmove
	// to avoid moving w to the heap.
	var w iword
	switch n {
	default:
		panic("reflect: internal error: loadIword of " + strconv.Itoa(int(n)) + "-byte value")
	case 0:
	case 1:
		*(*uint8)(unsafe.Pointer(&w)) = *(*uint8)(p)
	case 2:
		*(*uint16)(unsafe.Pointer(&w)) = *(*uint16)(p)
	case 3:
		*(*[3]byte)(unsafe.Pointer(&w)) = *(*[3]byte)(p)
	case 4:
		*(*uint32)(unsafe.Pointer(&w)) = *(*uint32)(p)
	case 5:
		*(*[5]byte)(unsafe.Pointer(&w)) = *(*[5]byte)(p)
	case 6:
		*(*[6]byte)(unsafe.Pointer(&w)) = *(*[6]byte)(p)
	case 7:
		*(*[7]byte)(unsafe.Pointer(&w)) = *(*[7]byte)(p)
	case 8:
		*(*uint64)(unsafe.Pointer(&w)) = *(*uint64)(p)
	}
	return w
}

// storeIword stores n bytes from w into p.
func storeIword(p unsafe.Pointer, w iword, n uintptr) {
	// Run the copy ourselves instead of calling memmove
	// to avoid moving w to the heap.
	switch n {
	default:
		panic("reflect: internal error: storeIword of " + strconv.Itoa(int(n)) + "-byte value")
	case 0:
	case 1:
		*(*uint8)(p) = *(*uint8)(unsafe.Pointer(&w))
	case 2:
		*(*uint16)(p) = *(*uint16)(unsafe.Pointer(&w))
	case 3:
		*(*[3]byte)(p) = *(*[3]byte)(unsafe.Pointer(&w))
	case 4:
		*(*uint32)(p) = *(*uint32)(unsafe.Pointer(&w))
	case 5:
		*(*[5]byte)(p) = *(*[5]byte)(unsafe.Pointer(&w))
	case 6:
		*(*[6]byte)(p) = *(*[6]byte)(unsafe.Pointer(&w))
	case 7:
		*(*[7]byte)(p) = *(*[7]byte)(unsafe.Pointer(&w))
	case 8:
		*(*uint64)(p) = *(*uint64)(unsafe.Pointer(&w))
	}
}

// emptyInterface is the header for an interface{} value.
type emptyInterface struct {
	typ  *rtype
	word iword
}

// nonEmptyInterface is the header for a interface value with methods.
type nonEmptyInterface struct {
	// see ../runtime/iface.c:/Itab
	itab *struct {
		ityp   *rtype // static interface type
		typ    *rtype // dynamic concrete type
		link   unsafe.Pointer
		bad    int32
		unused int32
		fun    [100000]unsafe.Pointer // method table
	}
	word iword
}

// mustBe panics if f's kind is not expected.
// Making this a method on flag instead of on Value
// (and embedding flag in Value) means that we can write
// the very clear v.mustBe(Bool) and have it compile into
// v.flag.mustBe(Bool), which will only bother to copy the
// single important word for the receiver.
func (f flag) mustBe(expected Kind) {
	k := f.kind()
	if k != expected {
		panic(&ValueError{methodName(), k})
	}
}

// mustBeExported panics if f records that the value was obtained using
// an unexported field.
func (f flag) mustBeExported() {
	if f == 0 {
		panic(&ValueError{methodName(), 0})
	}
	if f&flagRO != 0 {
		panic("reflect: " + methodName() + " using value obtained using unexported field")
	}
}

// mustBeAssignable panics if f records that the value is not assignable,
// which is to say that either it was obtained using an unexported field
// or it is not addressable.
func (f flag) mustBeAssignable() {
	if f == 0 {
		panic(&ValueError{methodName(), Invalid})
	}
	// Assignable if addressable and not read-only.
	if f&flagRO != 0 {
		panic("reflect: " + methodName() + " using value obtained using unexported field")
	}
	if f&flagAddr == 0 {
		panic("reflect: " + methodName() + " using unaddressable value")
	}
}

// Addr returns a pointer value representing the address of v.
// It panics if CanAddr() returns false.
// Addr is typically used to obtain a pointer to a struct field
// or slice element in order to call a method that requires a
// pointer receiver.
func (v Value) Addr() Value {
	if v.flag&flagAddr == 0 {
		panic("reflect.Value.Addr of unaddressable value")
	}
	return Value{v.typ.ptrTo(), v.val, (v.flag & flagRO) | flag(Ptr)<<flagKindShift}
}

// Bool returns v's underlying value.
// It panics if v's kind is not Bool.
func (v Value) Bool() bool {
	v.mustBe(Bool)
	if v.flag&flagIndir != 0 {
		return *(*bool)(v.val)
	}
	return *(*bool)(unsafe.Pointer(&v.val))
}

// Bytes returns v's underlying value.
// It panics if v's underlying value is not a slice of bytes.
func (v Value) Bytes() []byte {
	v.mustBe(Slice)
	if v.typ.Elem().Kind() != Uint8 {
		panic("reflect.Value.Bytes of non-byte slice")
	}
	// Slice is always bigger than a word; assume flagIndir.
	return *(*[]byte)(v.val)
}

// runes returns v's underlying value.
// It panics if v's underlying value is not a slice of runes (int32s).
func (v Value) runes() []rune {
	v.mustBe(Slice)
	if v.typ.Elem().Kind() != Int32 {
		panic("reflect.Value.Bytes of non-rune slice")
	}
	// Slice is always bigger than a word; assume flagIndir.
	return *(*[]rune)(v.val)
}

// CanAddr returns true if the value's address can be obtained with Addr.
// Such values are called addressable.  A value is addressable if it is
// an element of a slice, an element of an addressable array,
// a field of an addressable struct, or the result of dereferencing a pointer.
// If CanAddr returns false, calling Addr will panic.
func (v Value) CanAddr() bool {
	return v.flag&flagAddr != 0
}

// CanSet returns true if the value of v can be changed.
// A Value can be changed only if it is addressable and was not
// obtained by the use of unexported struct fields.
// If CanSet returns false, calling Set or any type-specific
// setter (e.g., SetBool, SetInt64) will panic.
func (v Value) CanSet() bool {
	return v.flag&(flagAddr|flagRO) == flagAddr
}

// Call calls the function v with the input arguments in.
// For example, if len(in) == 3, v.Call(in) represents the Go call v(in[0], in[1], in[2]).
// Call panics if v's Kind is not Func.
// It returns the output results as Values.
// As in Go, each input argument must be assignable to the
// type of the function's corresponding input parameter.
// If v is a variadic function, Call creates the variadic slice parameter
// itself, copying in the corresponding values.
func (v Value) Call(in []Value) []Value {
	v.mustBe(Func)
	v.mustBeExported()
	return v.call("Call", in)
}

// CallSlice calls the variadic function v with the input arguments in,
// assigning the slice in[len(in)-1] to v's final variadic argument.
// For example, if len(in) == 3, v.Call(in) represents the Go call v(in[0], in[1], in[2]...).
// Call panics if v's Kind is not Func or if v is not variadic.
// It returns the output results as Values.
// As in Go, each input argument must be assignable to the
// type of the function's corresponding input parameter.
func (v Value) CallSlice(in []Value) []Value {
	v.mustBe(Func)
	v.mustBeExported()
	return v.call("CallSlice", in)
}

func (v Value) call(op string, in []Value) []Value {
	// Get function pointer, type.
	t := v.typ
	var (
		fn   unsafe.Pointer
		rcvr iword
	)
	if v.flag&flagMethod != 0 {
		t, fn, rcvr = methodReceiver(op, v, int(v.flag)>>flagMethodShift)
	} else if v.flag&flagIndir != 0 {
		fn = *(*unsafe.Pointer)(v.val)
	} else {
		fn = v.val
	}

	if fn == nil {
		panic("reflect.Value.Call: call of nil function")
	}

	isSlice := op == "CallSlice"
	n := t.NumIn()
	if isSlice {
		if !t.IsVariadic() {
			panic("reflect: CallSlice of non-variadic function")
		}
		if len(in) < n {
			panic("reflect: CallSlice with too few input arguments")
		}
		if len(in) > n {
			panic("reflect: CallSlice with too many input arguments")
		}
	} else {
		if t.IsVariadic() {
			n--
		}
		if len(in) < n {
			panic("reflect: Call with too few input arguments")
		}
		if !t.IsVariadic() && len(in) > n {
			panic("reflect: Call with too many input arguments")
		}
	}
	for _, x := range in {
		if x.Kind() == Invalid {
			panic("reflect: " + op + " using zero Value argument")
		}
	}
	for i := 0; i < n; i++ {
		if xt, targ := in[i].Type(), t.In(i); !xt.AssignableTo(targ) {
			panic("reflect: " + op + " using " + xt.String() + " as type " + targ.String())
		}
	}
	if !isSlice && t.IsVariadic() {
		// prepare slice for remaining values
		m := len(in) - n
		slice := MakeSlice(t.In(n), m, m)
		elem := t.In(n).Elem()
		for i := 0; i < m; i++ {
			x := in[n+i]
			if xt := x.Type(); !xt.AssignableTo(elem) {
				panic("reflect: cannot use " + xt.String() + " as type " + elem.String() + " in " + op)
			}
			slice.Index(i).Set(x)
		}
		origIn := in
		in = make([]Value, n+1)
		copy(in[:n], origIn)
		in[n] = slice
	}

	nin := len(in)
	if nin != t.NumIn() {
		panic("reflect.Value.Call: wrong argument count")
	}
	nout := t.NumOut()

	// Compute arg size & allocate.
	// This computation is 5g/6g/8g-dependent
	// and probably wrong for gccgo, but so
	// is most of this function.
	size, _, _, _ := frameSize(t, v.flag&flagMethod != 0)

	// Copy into args.
	//
	// TODO(rsc): This will need to be updated for any new garbage collector.
	// For now make everything look like a pointer by allocating
	// a []unsafe.Pointer.
	args := make([]unsafe.Pointer, size/ptrSize)
	ptr := uintptr(unsafe.Pointer(&args[0]))
	off := uintptr(0)
	if v.flag&flagMethod != 0 {
		// Hard-wired first argument.
		*(*iword)(unsafe.Pointer(ptr)) = rcvr
		off = ptrSize
	}
	for i, v := range in {
		v.mustBeExported()
		targ := t.In(i).(*rtype)
		a := uintptr(targ.align)
		off = (off + a - 1) &^ (a - 1)
		n := targ.size
		addr := unsafe.Pointer(ptr + off)
		v = v.assignTo("reflect.Value.Call", targ, (*interface{})(addr))
		if v.flag&flagIndir == 0 {
			storeIword(addr, iword(v.val), n)
		} else {
			memmove(addr, v.val, n)
		}
		off += n
	}
	off = (off + ptrSize - 1) &^ (ptrSize - 1)

	// Call.
	call(fn, unsafe.Pointer(ptr), uint32(size))

	// Copy return values out of args.
	//
	// TODO(rsc): revisit like above.
	ret := make([]Value, nout)
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		a := uintptr(tv.Align())
		off = (off + a - 1) &^ (a - 1)
		fl := flagIndir | flag(tv.Kind())<<flagKindShift
		ret[i] = Value{tv.common(), unsafe.Pointer(ptr + off), fl}
		off += tv.Size()
	}

	return ret
}

// callReflect is the call implementation used by a function
// returned by MakeFunc. In many ways it is the opposite of the
// method Value.call above. The method above converts a call using Values
// into a call of a function with a concrete argument frame, while
// callReflect converts a call of a function with a concrete argument
// frame into a call using Values.
// It is in this file so that it can be next to the call method above.
// The remainder of the MakeFunc implementation is in makefunc.go.
func callReflect(ctxt *makeFuncImpl, frame unsafe.Pointer) {
	ftyp := ctxt.typ
	f := ctxt.fn

	// Copy argument frame into Values.
	ptr := frame
	off := uintptr(0)
	in := make([]Value, 0, len(ftyp.in))
	for _, arg := range ftyp.in {
		typ := arg
		off += -off & uintptr(typ.align-1)
		v := Value{typ, nil, flag(typ.Kind()) << flagKindShift}
		if typ.size <= ptrSize {
			// value fits in word.
			v.val = unsafe.Pointer(loadIword(unsafe.Pointer(uintptr(ptr)+off), typ.size))
		} else {
			// value does not fit in word.
			// Must make a copy, because f might keep a reference to it,
			// and we cannot let f keep a reference to the stack frame
			// after this function returns, not even a read-only reference.
			v.val = unsafe_New(typ)
			memmove(v.val, unsafe.Pointer(uintptr(ptr)+off), typ.size)
			v.flag |= flagIndir
		}
		in = append(in, v)
		off += typ.size
	}

	// Call underlying function.
	out := f(in)
	if len(out) != len(ftyp.out) {
		panic("reflect: wrong return count from function created by MakeFunc")
	}

	// Copy results back into argument frame.
	if len(ftyp.out) > 0 {
		off += -off & (ptrSize - 1)
		for i, arg := range ftyp.out {
			typ := arg
			v := out[i]
			if v.typ != typ {
				panic("reflect: function created by MakeFunc using " + funcName(f) +
					" returned wrong type: have " +
					out[i].typ.String() + " for " + typ.String())
			}
			if v.flag&flagRO != 0 {
				panic("reflect: function created by MakeFunc using " + funcName(f) +
					" returned value obtained from unexported field")
			}
			off += -off & uintptr(typ.align-1)
			addr := unsafe.Pointer(uintptr(ptr) + off)
			if v.flag&flagIndir == 0 {
				storeIword(addr, iword(v.val), typ.size)
			} else {
				memmove(addr, v.val, typ.size)
			}
			off += typ.size
		}
	}
}

// methodReceiver returns information about the receiver
// described by v. The Value v may or may not have the
// flagMethod bit set, so the kind cached in v.flag should
// not be used.
func methodReceiver(op string, v Value, methodIndex int) (t *rtype, fn unsafe.Pointer, rcvr iword) {
	i := methodIndex
	if v.typ.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(v.typ))
		if i < 0 || i >= len(tt.methods) {
			panic("reflect: internal error: invalid method index")
		}
		m := &tt.methods[i]
		if m.pkgPath != nil {
			panic("reflect: " + op + " of unexported method")
		}
		t = m.typ
		iface := (*nonEmptyInterface)(v.val)
		if iface.itab == nil {
			panic("reflect: " + op + " of method on nil interface value")
		}
		fn = unsafe.Pointer(&iface.itab.fun[i])
		rcvr = iface.word
	} else {
		ut := v.typ.uncommon()
		if ut == nil || i < 0 || i >= len(ut.methods) {
			panic("reflect: internal error: invalid method index")
		}
		m := &ut.methods[i]
		if m.pkgPath != nil {
			panic("reflect: " + op + " of unexported method")
		}
		fn = unsafe.Pointer(&m.ifn)
		t = m.mtyp
		rcvr = v.iword()
	}
	return
}

// align returns the result of rounding x up to a multiple of n.
// n must be a power of two.
func align(x, n uintptr) uintptr {
	return (x + n - 1) &^ (n - 1)
}

// frameSize returns the sizes of the argument and result frame
// for a function of the given type. The rcvr bool specifies whether
// a one-word receiver should be included in the total.
func frameSize(t *rtype, rcvr bool) (total, in, outOffset, out uintptr) {
	if rcvr {
		// extra word for receiver interface word
		total += ptrSize
	}

	nin := t.NumIn()
	in = -total
	for i := 0; i < nin; i++ {
		tv := t.In(i)
		total = align(total, uintptr(tv.Align()))
		total += tv.Size()
	}
	in += total
	total = align(total, ptrSize)
	nout := t.NumOut()
	outOffset = total
	out = -total
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		total = align(total, uintptr(tv.Align()))
		total += tv.Size()
	}
	out += total

	// total must be > 0 in order for &args[0] to be valid.
	// the argument copying is going to round it up to
	// a multiple of ptrSize anyway, so make it ptrSize to begin with.
	if total < ptrSize {
		total = ptrSize
	}

	// round to pointer
	total = align(total, ptrSize)

	return
}

// callMethod is the call implementation used by a function returned
// by makeMethodValue (used by v.Method(i).Interface()).
// It is a streamlined version of the usual reflect call: the caller has
// already laid out the argument frame for us, so we don't have
// to deal with individual Values for each argument.
// It is in this file so that it can be next to the two similar functions above.
// The remainder of the makeMethodValue implementation is in makefunc.go.
func callMethod(ctxt *methodValue, frame unsafe.Pointer) {
	t, fn, rcvr := methodReceiver("call", ctxt.rcvr, ctxt.method)
	total, in, outOffset, out := frameSize(t, true)

	// Copy into args.
	//
	// TODO(rsc): This will need to be updated for any new garbage collector.
	// For now make everything look like a pointer by allocating
	// a []unsafe.Pointer.
	args := make([]unsafe.Pointer, total/ptrSize)
	args[0] = unsafe.Pointer(rcvr)
	base := unsafe.Pointer(&args[0])
	memmove(unsafe.Pointer(uintptr(base)+ptrSize), frame, in)

	// Call.
	call(fn, unsafe.Pointer(&args[0]), uint32(total))

	// Copy return values.
	memmove(unsafe.Pointer(uintptr(frame)+outOffset-ptrSize), unsafe.Pointer(uintptr(base)+outOffset), out)
}

// funcName returns the name of f, for use in error messages.
func funcName(f func([]Value) []Value) string {
	pc := *(*uintptr)(unsafe.Pointer(&f))
	rf := runtime.FuncForPC(pc)
	if rf != nil {
		return rf.Name()
	}
	return "closure"
}

// Cap returns v's capacity.
// It panics if v's Kind is not Array, Chan, or Slice.
func (v Value) Cap() int {
	k := v.kind()
	switch k {
	case Array:
		return v.typ.Len()
	case Chan:
		return int(chancap(v.iword()))
	case Slice:
		// Slice is always bigger than a word; assume flagIndir.
		return (*SliceHeader)(v.val).Cap
	}
	panic(&ValueError{"reflect.Value.Cap", k})
}

// Close closes the channel v.
// It panics if v's Kind is not Chan.
func (v Value) Close() {
	v.mustBe(Chan)
	v.mustBeExported()
	chanclose(v.iword())
}

// Complex returns v's underlying value, as a complex128.
// It panics if v's Kind is not Complex64 or Complex128
func (v Value) Complex() complex128 {
	k := v.kind()
	switch k {
	case Complex64:
		if v.flag&flagIndir != 0 {
			return complex128(*(*complex64)(v.val))
		}
		return complex128(*(*complex64)(unsafe.Pointer(&v.val)))
	case Complex128:
		// complex128 is always bigger than a word; assume flagIndir.
		return *(*complex128)(v.val)
	}
	panic(&ValueError{"reflect.Value.Complex", k})
}

// Elem returns the value that the interface v contains
// or that the pointer v points to.
// It panics if v's Kind is not Interface or Ptr.
// It returns the zero Value if v is nil.
func (v Value) Elem() Value {
	k := v.kind()
	switch k {
	case Interface:
		var (
			typ *rtype
			val unsafe.Pointer
		)
		if v.typ.NumMethod() == 0 {
			eface := (*emptyInterface)(v.val)
			if eface.typ == nil {
				// nil interface value
				return Value{}
			}
			typ = eface.typ
			val = unsafe.Pointer(eface.word)
		} else {
			iface := (*nonEmptyInterface)(v.val)
			if iface.itab == nil {
				// nil interface value
				return Value{}
			}
			typ = iface.itab.typ
			val = unsafe.Pointer(iface.word)
		}
		fl := v.flag & flagRO
		fl |= flag(typ.Kind()) << flagKindShift
		if typ.size > ptrSize {
			fl |= flagIndir
		}
		return Value{typ, val, fl}

	case Ptr:
		val := v.val
		if v.flag&flagIndir != 0 {
			val = *(*unsafe.Pointer)(val)
		}
		// The returned value's address is v's value.
		if val == nil {
			return Value{}
		}
		tt := (*ptrType)(unsafe.Pointer(v.typ))
		typ := tt.elem
		fl := v.flag&flagRO | flagIndir | flagAddr
		fl |= flag(typ.Kind() << flagKindShift)
		return Value{typ, val, fl}
	}
	panic(&ValueError{"reflect.Value.Elem", k})
}

// Field returns the i'th field of the struct v.
// It panics if v's Kind is not Struct or i is out of range.
func (v Value) Field(i int) Value {
	v.mustBe(Struct)
	tt := (*structType)(unsafe.Pointer(v.typ))
	if i < 0 || i >= len(tt.fields) {
		panic("reflect: Field index out of range")
	}
	field := &tt.fields[i]
	typ := field.typ

	// Inherit permission bits from v.
	fl := v.flag & (flagRO | flagIndir | flagAddr)
	// Using an unexported field forces flagRO.
	if field.pkgPath != nil {
		fl |= flagRO
	}
	fl |= flag(typ.Kind()) << flagKindShift

	var val unsafe.Pointer
	switch {
	case fl&flagIndir != 0:
		// Indirect.  Just bump pointer.
		val = unsafe.Pointer(uintptr(v.val) + field.offset)
	case bigEndian:
		// Direct.  Discard leading bytes.
		val = unsafe.Pointer(uintptr(v.val) << (field.offset * 8))
	default:
		// Direct.  Discard leading bytes.
		val = unsafe.Pointer(uintptr(v.val) >> (field.offset * 8))
	}

	return Value{typ, val, fl}
}

// FieldByIndex returns the nested field corresponding to index.
// It panics if v's Kind is not struct.
func (v Value) FieldByIndex(index []int) Value {
	v.mustBe(Struct)
	for i, x := range index {
		if i > 0 {
			if v.Kind() == Ptr && v.Elem().Kind() == Struct {
				v = v.Elem()
			}
		}
		v = v.Field(x)
	}
	return v
}

// FieldByName returns the struct field with the given name.
// It returns the zero Value if no field was found.
// It panics if v's Kind is not struct.
func (v Value) FieldByName(name string) Value {
	v.mustBe(Struct)
	if f, ok := v.typ.FieldByName(name); ok {
		return v.FieldByIndex(f.Index)
	}
	return Value{}
}

// FieldByNameFunc returns the struct field with a name
// that satisfies the match function.
// It panics if v's Kind is not struct.
// It returns the zero Value if no field was found.
func (v Value) FieldByNameFunc(match func(string) bool) Value {
	v.mustBe(Struct)
	if f, ok := v.typ.FieldByNameFunc(match); ok {
		return v.FieldByIndex(f.Index)
	}
	return Value{}
}

// Float returns v's underlying value, as a float64.
// It panics if v's Kind is not Float32 or Float64
func (v Value) Float() float64 {
	k := v.kind()
	switch k {
	case Float32:
		if v.flag&flagIndir != 0 {
			return float64(*(*float32)(v.val))
		}
		return float64(*(*float32)(unsafe.Pointer(&v.val)))
	case Float64:
		if v.flag&flagIndir != 0 {
			return *(*float64)(v.val)
		}
		return *(*float64)(unsafe.Pointer(&v.val))
	}
	panic(&ValueError{"reflect.Value.Float", k})
}

var uint8Type = TypeOf(uint8(0)).(*rtype)

// Index returns v's i'th element.
// It panics if v's Kind is not Array, Slice, or String or i is out of range.
func (v Value) Index(i int) Value {
	k := v.kind()
	switch k {
	case Array:
		tt := (*arrayType)(unsafe.Pointer(v.typ))
		if i < 0 || i > int(tt.len) {
			panic("reflect: array index out of range")
		}
		typ := tt.elem
		fl := v.flag & (flagRO | flagIndir | flagAddr) // bits same as overall array
		fl |= flag(typ.Kind()) << flagKindShift
		offset := uintptr(i) * typ.size

		var val unsafe.Pointer
		switch {
		case fl&flagIndir != 0:
			// Indirect.  Just bump pointer.
			val = unsafe.Pointer(uintptr(v.val) + offset)
		case bigEndian:
			// Direct.  Discard leading bytes.
			val = unsafe.Pointer(uintptr(v.val) << (offset * 8))
		default:
			// Direct.  Discard leading bytes.
			val = unsafe.Pointer(uintptr(v.val) >> (offset * 8))
		}
		return Value{typ, val, fl}

	case Slice:
		// Element flag same as Elem of Ptr.
		// Addressable, indirect, possibly read-only.
		fl := flagAddr | flagIndir | v.flag&flagRO
		s := (*SliceHeader)(v.val)
		if i < 0 || i >= s.Len {
			panic("reflect: slice index out of range")
		}
		tt := (*sliceType)(unsafe.Pointer(v.typ))
		typ := tt.elem
		fl |= flag(typ.Kind()) << flagKindShift
		val := unsafe.Pointer(s.Data + uintptr(i)*typ.size)
		return Value{typ, val, fl}

	case String:
		fl := v.flag&flagRO | flag(Uint8<<flagKindShift)
		s := (*StringHeader)(v.val)
		if i < 0 || i >= s.Len {
			panic("reflect: string index out of range")
		}
		val := *(*byte)(unsafe.Pointer(s.Data + uintptr(i)))
		return Value{uint8Type, unsafe.Pointer(uintptr(val)), fl}
	}
	panic(&ValueError{"reflect.Value.Index", k})
}

// Int returns v's underlying value, as an int64.
// It panics if v's Kind is not Int, Int8, Int16, Int32, or Int64.
func (v Value) Int() int64 {
	k := v.kind()
	var p unsafe.Pointer
	if v.flag&flagIndir != 0 {
		p = v.val
	} else {
		// The escape analysis is good enough that &v.val
		// does not trigger a heap allocation.
		p = unsafe.Pointer(&v.val)
	}
	switch k {
	case Int:
		return int64(*(*int)(p))
	case Int8:
		return int64(*(*int8)(p))
	case Int16:
		return int64(*(*int16)(p))
	case Int32:
		return int64(*(*int32)(p))
	case Int64:
		return int64(*(*int64)(p))
	}
	panic(&ValueError{"reflect.Value.Int", k})
}

// CanInterface returns true if Interface can be used without panicking.
func (v Value) CanInterface() bool {
	if v.flag == 0 {
		panic(&ValueError{"reflect.Value.CanInterface", Invalid})
	}
	return v.flag&flagRO == 0
}

// Interface returns v's current value as an interface{}.
// It is equivalent to:
//	var i interface{} = (v's underlying value)
// If v is a method obtained by invoking Value.Method
// (as opposed to Type.Method), Interface cannot return an
// interface value, so it panics.
// It also panics if the Value was obtained by accessing
// unexported struct fields.
func (v Value) Interface() (i interface{}) {
	return valueInterface(v, true)
}

func valueInterface(v Value, safe bool) interface{} {
	if v.flag == 0 {
		panic(&ValueError{"reflect.Value.Interface", 0})
	}
	if safe && v.flag&flagRO != 0 {
		// Do not allow access to unexported values via Interface,
		// because they might be pointers that should not be
		// writable or methods or function that should not be callable.
		panic("reflect.Value.Interface: cannot return value obtained from unexported field or method")
	}
	if v.flag&flagMethod != 0 {
		v = makeMethodValue("Interface", v)
	}

	k := v.kind()
	if k == Interface {
		// Special case: return the element inside the interface.
		// Empty interface has one layout, all interfaces with
		// methods have a second layout.
		if v.NumMethod() == 0 {
			return *(*interface{})(v.val)
		}
		return *(*interface {
			M()
		})(v.val)
	}

	// Non-interface value.
	var eface emptyInterface
	eface.typ = v.typ
	eface.word = v.iword()

	if v.flag&flagIndir != 0 && v.typ.size > ptrSize {
		// eface.word is a pointer to the actual data,
		// which might be changed.  We need to return
		// a pointer to unchanging data, so make a copy.
		ptr := unsafe_New(v.typ)
		memmove(ptr, unsafe.Pointer(eface.word), v.typ.size)
		eface.word = iword(ptr)
	}

	return *(*interface{})(unsafe.Pointer(&eface))
}

// InterfaceData returns the interface v's value as a uintptr pair.
// It panics if v's Kind is not Interface.
func (v Value) InterfaceData() [2]uintptr {
	v.mustBe(Interface)
	// We treat this as a read operation, so we allow
	// it even for unexported data, because the caller
	// has to import "unsafe" to turn it into something
	// that can be abused.
	// Interface value is always bigger than a word; assume flagIndir.
	return *(*[2]uintptr)(v.val)
}

// IsNil returns true if v is a nil value.
// It panics if v's Kind is not Chan, Func, Interface, Map, Ptr, or Slice.
func (v Value) IsNil() bool {
	k := v.kind()
	switch k {
	case Chan, Func, Map, Ptr:
		if v.flag&flagMethod != 0 {
			return false
		}
		ptr := v.val
		if v.flag&flagIndir != 0 {
			ptr = *(*unsafe.Pointer)(ptr)
		}
		return ptr == nil
	case Interface, Slice:
		// Both interface and slice are nil if first word is 0.
		// Both are always bigger than a word; assume flagIndir.
		return *(*unsafe.Pointer)(v.val) == nil
	}
	panic(&ValueError{"reflect.Value.IsNil", k})
}

// IsValid returns true if v represents a value.
// It returns false if v is the zero Value.
// If IsValid returns false, all other methods except String panic.
// Most functions and methods never return an invalid value.
// If one does, its documentation states the conditions explicitly.
func (v Value) IsValid() bool {
	return v.flag != 0
}

// Kind returns v's Kind.
// If v is the zero Value (IsValid returns false), Kind returns Invalid.
func (v Value) Kind() Kind {
	return v.kind()
}

// Len returns v's length.
// It panics if v's Kind is not Array, Chan, Map, Slice, or String.
func (v Value) Len() int {
	k := v.kind()
	switch k {
	case Array:
		tt := (*arrayType)(unsafe.Pointer(v.typ))
		return int(tt.len)
	case Chan:
		return chanlen(v.iword())
	case Map:
		return maplen(v.iword())
	case Slice:
		// Slice is bigger than a word; assume flagIndir.
		return (*SliceHeader)(v.val).Len
	case String:
		// String is bigger than a word; assume flagIndir.
		return (*StringHeader)(v.val).Len
	}
	panic(&ValueError{"reflect.Value.Len", k})
}

// MapIndex returns the value associated with key in the map v.
// It panics if v's Kind is not Map.
// It returns the zero Value if key is not found in the map or if v represents a nil map.
// As in Go, the key's value must be assignable to the map's key type.
func (v Value) MapIndex(key Value) Value {
	v.mustBe(Map)
	tt := (*mapType)(unsafe.Pointer(v.typ))

	// Do not require key to be exported, so that DeepEqual
	// and other programs can use all the keys returned by
	// MapKeys as arguments to MapIndex.  If either the map
	// or the key is unexported, though, the result will be
	// considered unexported.  This is consistent with the
	// behavior for structs, which allow read but not write
	// of unexported fields.
	key = key.assignTo("reflect.Value.MapIndex", tt.key, nil)

	word, ok := mapaccess(v.typ, v.iword(), key.iword())
	if !ok {
		return Value{}
	}
	typ := tt.elem
	fl := (v.flag | key.flag) & flagRO
	if typ.size > ptrSize {
		fl |= flagIndir
	}
	fl |= flag(typ.Kind()) << flagKindShift
	return Value{typ, unsafe.Pointer(word), fl}
}

// MapKeys returns a slice containing all the keys present in the map,
// in unspecified order.
// It panics if v's Kind is not Map.
// It returns an empty slice if v represents a nil map.
func (v Value) MapKeys() []Value {
	v.mustBe(Map)
	tt := (*mapType)(unsafe.Pointer(v.typ))
	keyType := tt.key

	fl := v.flag & flagRO
	fl |= flag(keyType.Kind()) << flagKindShift
	if keyType.size > ptrSize {
		fl |= flagIndir
	}

	m := v.iword()
	mlen := int(0)
	if m != nil {
		mlen = maplen(m)
	}
	it := mapiterinit(v.typ, m)
	a := make([]Value, mlen)
	var i int
	for i = 0; i < len(a); i++ {
		keyWord, ok := mapiterkey(it)
		if !ok {
			break
		}
		a[i] = Value{keyType, unsafe.Pointer(keyWord), fl}
		mapiternext(it)
	}
	return a[:i]
}

// Method returns a function value corresponding to v's i'th method.
// The arguments to a Call on the returned function should not include
// a receiver; the returned function will always use v as the receiver.
// Method panics if i is out of range or if v is a nil interface value.
func (v Value) Method(i int) Value {
	if v.typ == nil {
		panic(&ValueError{"reflect.Value.Method", Invalid})
	}
	if v.flag&flagMethod != 0 || i < 0 || i >= v.typ.NumMethod() {
		panic("reflect: Method index out of range")
	}
	if v.typ.Kind() == Interface && v.IsNil() {
		panic("reflect: Method on nil interface value")
	}
	fl := v.flag & (flagRO | flagIndir)
	fl |= flag(Func) << flagKindShift
	fl |= flag(i)<<flagMethodShift | flagMethod
	return Value{v.typ, v.val, fl}
}

// NumMethod returns the number of methods in the value's method set.
func (v Value) NumMethod() int {
	if v.typ == nil {
		panic(&ValueError{"reflect.Value.NumMethod", Invalid})
	}
	if v.flag&flagMethod != 0 {
		return 0
	}
	return v.typ.NumMethod()
}

// MethodByName returns a function value corresponding to the method
// of v with the given name.
// The arguments to a Call on the returned function should not include
// a receiver; the returned function will always use v as the receiver.
// It returns the zero Value if no method was found.
func (v Value) MethodByName(name string) Value {
	if v.typ == nil {
		panic(&ValueError{"reflect.Value.MethodByName", Invalid})
	}
	if v.flag&flagMethod != 0 {
		return Value{}
	}
	m, ok := v.typ.MethodByName(name)
	if !ok {
		return Value{}
	}
	return v.Method(m.Index)
}

// NumField returns the number of fields in the struct v.
// It panics if v's Kind is not Struct.
func (v Value) NumField() int {
	v.mustBe(Struct)
	tt := (*structType)(unsafe.Pointer(v.typ))
	return len(tt.fields)
}

// OverflowComplex returns true if the complex128 x cannot be represented by v's type.
// It panics if v's Kind is not Complex64 or Complex128.
func (v Value) OverflowComplex(x complex128) bool {
	k := v.kind()
	switch k {
	case Complex64:
		return overflowFloat32(real(x)) || overflowFloat32(imag(x))
	case Complex128:
		return false
	}
	panic(&ValueError{"reflect.Value.OverflowComplex", k})
}

// OverflowFloat returns true if the float64 x cannot be represented by v's type.
// It panics if v's Kind is not Float32 or Float64.
func (v Value) OverflowFloat(x float64) bool {
	k := v.kind()
	switch k {
	case Float32:
		return overflowFloat32(x)
	case Float64:
		return false
	}
	panic(&ValueError{"reflect.Value.OverflowFloat", k})
}

func overflowFloat32(x float64) bool {
	if x < 0 {
		x = -x
	}
	return math.MaxFloat32 < x && x <= math.MaxFloat64
}

// OverflowInt returns true if the int64 x cannot be represented by v's type.
// It panics if v's Kind is not Int, Int8, int16, Int32, or Int64.
func (v Value) OverflowInt(x int64) bool {
	k := v.kind()
	switch k {
	case Int, Int8, Int16, Int32, Int64:
		bitSize := v.typ.size * 8
		trunc := (x << (64 - bitSize)) >> (64 - bitSize)
		return x != trunc
	}
	panic(&ValueError{"reflect.Value.OverflowInt", k})
}

// OverflowUint returns true if the uint64 x cannot be represented by v's type.
// It panics if v's Kind is not Uint, Uintptr, Uint8, Uint16, Uint32, or Uint64.
func (v Value) OverflowUint(x uint64) bool {
	k := v.kind()
	switch k {
	case Uint, Uintptr, Uint8, Uint16, Uint32, Uint64:
		bitSize := v.typ.size * 8
		trunc := (x << (64 - bitSize)) >> (64 - bitSize)
		return x != trunc
	}
	panic(&ValueError{"reflect.Value.OverflowUint", k})
}

// Pointer returns v's value as a uintptr.
// It returns uintptr instead of unsafe.Pointer so that
// code using reflect cannot obtain unsafe.Pointers
// without importing the unsafe package explicitly.
// It panics if v's Kind is not Chan, Func, Map, Ptr, Slice, or UnsafePointer.
//
// If v's Kind is Func, the returned pointer is an underlying
// code pointer, but not necessarily enough to identify a
// single function uniquely. The only guarantee is that the
// result is zero if and only if v is a nil func Value.
func (v Value) Pointer() uintptr {
	k := v.kind()
	switch k {
	case Chan, Map, Ptr, UnsafePointer:
		p := v.val
		if v.flag&flagIndir != 0 {
			p = *(*unsafe.Pointer)(p)
		}
		return uintptr(p)
	case Func:
		if v.flag&flagMethod != 0 {
			// As the doc comment says, the returned pointer is an
			// underlying code pointer but not necessarily enough to
			// identify a single function uniquely. All method expressions
			// created via reflect have the same underlying code pointer,
			// so their Pointers are equal. The function used here must
			// match the one used in makeMethodValue.
			f := methodValueCall
			return **(**uintptr)(unsafe.Pointer(&f))
		}
		p := v.val
		if v.flag&flagIndir != 0 {
			p = *(*unsafe.Pointer)(p)
		}
		// Non-nil func value points at data block.
		// First word of data block is actual code.
		if p != nil {
			p = *(*unsafe.Pointer)(p)
		}
		return uintptr(p)

	case Slice:
		return (*SliceHeader)(v.val).Data
	}
	panic(&ValueError{"reflect.Value.Pointer", k})
}

// Recv receives and returns a value from the channel v.
// It panics if v's Kind is not Chan.
// The receive blocks until a value is ready.
// The boolean value ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v Value) Recv() (x Value, ok bool) {
	v.mustBe(Chan)
	v.mustBeExported()
	return v.recv(false)
}

// internal recv, possibly non-blocking (nb).
// v is known to be a channel.
func (v Value) recv(nb bool) (val Value, ok bool) {
	tt := (*chanType)(unsafe.Pointer(v.typ))
	if ChanDir(tt.dir)&RecvDir == 0 {
		panic("reflect: recv on send-only channel")
	}
	word, selected, ok := chanrecv(v.typ, v.iword(), nb)
	if selected {
		typ := tt.elem
		fl := flag(typ.Kind()) << flagKindShift
		if typ.size > ptrSize {
			fl |= flagIndir
		}
		val = Value{typ, unsafe.Pointer(word), fl}
	}
	return
}

// Send sends x on the channel v.
// It panics if v's kind is not Chan or if x's type is not the same type as v's element type.
// As in Go, x's value must be assignable to the channel's element type.
func (v Value) Send(x Value) {
	v.mustBe(Chan)
	v.mustBeExported()
	v.send(x, false)
}

// internal send, possibly non-blocking.
// v is known to be a channel.
func (v Value) send(x Value, nb bool) (selected bool) {
	tt := (*chanType)(unsafe.Pointer(v.typ))
	if ChanDir(tt.dir)&SendDir == 0 {
		panic("reflect: send on recv-only channel")
	}
	x.mustBeExported()
	x = x.assignTo("reflect.Value.Send", tt.elem, nil)
	return chansend(v.typ, v.iword(), x.iword(), nb)
}

// Set assigns x to the value v.
// It panics if CanSet returns false.
// As in Go, x's value must be assignable to v's type.
func (v Value) Set(x Value) {
	v.mustBeAssignable()
	x.mustBeExported() // do not let unexported x leak
	var target *interface{}
	if v.kind() == Interface {
		target = (*interface{})(v.val)
	}
	x = x.assignTo("reflect.Set", v.typ, target)
	if x.flag&flagIndir != 0 {
		memmove(v.val, x.val, v.typ.size)
	} else {
		storeIword(v.val, iword(x.val), v.typ.size)
	}
}

// SetBool sets v's underlying value.
// It panics if v's Kind is not Bool or if CanSet() is false.
func (v Value) SetBool(x bool) {
	v.mustBeAssignable()
	v.mustBe(Bool)
	*(*bool)(v.val) = x
}

// SetBytes sets v's underlying value.
// It panics if v's underlying value is not a slice of bytes.
func (v Value) SetBytes(x []byte) {
	v.mustBeAssignable()
	v.mustBe(Slice)
	if v.typ.Elem().Kind() != Uint8 {
		panic("reflect.Value.SetBytes of non-byte slice")
	}
	*(*[]byte)(v.val) = x
}

// setRunes sets v's underlying value.
// It panics if v's underlying value is not a slice of runes (int32s).
func (v Value) setRunes(x []rune) {
	v.mustBeAssignable()
	v.mustBe(Slice)
	if v.typ.Elem().Kind() != Int32 {
		panic("reflect.Value.setRunes of non-rune slice")
	}
	*(*[]rune)(v.val) = x
}

// SetComplex sets v's underlying value to x.
// It panics if v's Kind is not Complex64 or Complex128, or if CanSet() is false.
func (v Value) SetComplex(x complex128) {
	v.mustBeAssignable()
	switch k := v.kind(); k {
	default:
		panic(&ValueError{"reflect.Value.SetComplex", k})
	case Complex64:
		*(*complex64)(v.val) = complex64(x)
	case Complex128:
		*(*complex128)(v.val) = x
	}
}

// SetFloat sets v's underlying value to x.
// It panics if v's Kind is not Float32 or Float64, or if CanSet() is false.
func (v Value) SetFloat(x float64) {
	v.mustBeAssignable()
	switch k := v.kind(); k {
	default:
		panic(&ValueError{"reflect.Value.SetFloat", k})
	case Float32:
		*(*float32)(v.val) = float32(x)
	case Float64:
		*(*float64)(v.val) = x
	}
}

// SetInt sets v's underlying value to x.
// It panics if v's Kind is not Int, Int8, Int16, Int32, or Int64, or if CanSet() is false.
func (v Value) SetInt(x int64) {
	v.mustBeAssignable()
	switch k := v.kind(); k {
	default:
		panic(&ValueError{"reflect.Value.SetInt", k})
	case Int:
		*(*int)(v.val) = int(x)
	case Int8:
		*(*int8)(v.val) = int8(x)
	case Int16:
		*(*int16)(v.val) = int16(x)
	case Int32:
		*(*int32)(v.val) = int32(x)
	case Int64:
		*(*int64)(v.val) = x
	}
}

// SetLen sets v's length to n.
// It panics if v's Kind is not Slice or if n is negative or
// greater than the capacity of the slice.
func (v Value) SetLen(n int) {
	v.mustBeAssignable()
	v.mustBe(Slice)
	s := (*SliceHeader)(v.val)
	if n < 0 || n > int(s.Cap) {
		panic("reflect: slice length out of range in SetLen")
	}
	s.Len = n
}

// SetMapIndex sets the value associated with key in the map v to val.
// It panics if v's Kind is not Map.
// If val is the zero Value, SetMapIndex deletes the key from the map.
// As in Go, key's value must be assignable to the map's key type,
// and val's value must be assignable to the map's value type.
func (v Value) SetMapIndex(key, val Value) {
	v.mustBe(Map)
	v.mustBeExported()
	key.mustBeExported()
	tt := (*mapType)(unsafe.Pointer(v.typ))
	key = key.assignTo("reflect.Value.SetMapIndex", tt.key, nil)
	if val.typ != nil {
		val.mustBeExported()
		val = val.assignTo("reflect.Value.SetMapIndex", tt.elem, nil)
	}
	mapassign(v.typ, v.iword(), key.iword(), val.iword(), val.typ != nil)
}

// SetUint sets v's underlying value to x.
// It panics if v's Kind is not Uint, Uintptr, Uint8, Uint16, Uint32, or Uint64, or if CanSet() is false.
func (v Value) SetUint(x uint64) {
	v.mustBeAssignable()
	switch k := v.kind(); k {
	default:
		panic(&ValueError{"reflect.Value.SetUint", k})
	case Uint:
		*(*uint)(v.val) = uint(x)
	case Uint8:
		*(*uint8)(v.val) = uint8(x)
	case Uint16:
		*(*uint16)(v.val) = uint16(x)
	case Uint32:
		*(*uint32)(v.val) = uint32(x)
	case Uint64:
		*(*uint64)(v.val) = x
	case Uintptr:
		*(*uintptr)(v.val) = uintptr(x)
	}
}

// SetPointer sets the unsafe.Pointer value v to x.
// It panics if v's Kind is not UnsafePointer.
func (v Value) SetPointer(x unsafe.Pointer) {
	v.mustBeAssignable()
	v.mustBe(UnsafePointer)
	*(*unsafe.Pointer)(v.val) = x
}

// SetString sets v's underlying value to x.
// It panics if v's Kind is not String or if CanSet() is false.
func (v Value) SetString(x string) {
	v.mustBeAssignable()
	v.mustBe(String)
	*(*string)(v.val) = x
}

// Slice returns a slice of v.
// It panics if v's Kind is not Array, Slice or String, or if v is an unaddressable array.
func (v Value) Slice(beg, end int) Value {
	var (
		cap  int
		typ  *sliceType
		base unsafe.Pointer
	)
	switch k := v.kind(); k {
	default:
		panic(&ValueError{"reflect.Value.Slice", k})

	case Array:
		if v.flag&flagAddr == 0 {
			panic("reflect.Value.Slice: slice of unaddressable array")
		}
		tt := (*arrayType)(unsafe.Pointer(v.typ))
		cap = int(tt.len)
		typ = (*sliceType)(unsafe.Pointer(tt.slice))
		base = v.val

	case Slice:
		typ = (*sliceType)(unsafe.Pointer(v.typ))
		s := (*SliceHeader)(v.val)
		base = unsafe.Pointer(s.Data)
		cap = s.Cap

	case String:
		s := (*StringHeader)(v.val)
		if beg < 0 || end < beg || end > s.Len {
			panic("reflect.Value.Slice: string slice index out of bounds")
		}
		var x string
		val := (*StringHeader)(unsafe.Pointer(&x))
		val.Data = s.Data + uintptr(beg)
		val.Len = end - beg
		return Value{v.typ, unsafe.Pointer(&x), v.flag}
	}

	if beg < 0 || end < beg || end > cap {
		panic("reflect.Value.Slice: slice index out of bounds")
	}

	// Declare slice so that gc can see the base pointer in it.
	var x []unsafe.Pointer

	// Reinterpret as *SliceHeader to edit.
	s := (*SliceHeader)(unsafe.Pointer(&x))
	s.Data = uintptr(base) + uintptr(beg)*typ.elem.Size()
	s.Len = end - beg
	s.Cap = cap - beg

	fl := v.flag&flagRO | flagIndir | flag(Slice)<<flagKindShift
	return Value{typ.common(), unsafe.Pointer(&x), fl}
}

// String returns the string v's underlying value, as a string.
// String is a special case because of Go's String method convention.
// Unlike the other getters, it does not panic if v's Kind is not String.
// Instead, it returns a string of the form "<T value>" where T is v's type.
func (v Value) String() string {
	switch k := v.kind(); k {
	case Invalid:
		return "<invalid Value>"
	case String:
		return *(*string)(v.val)
	}
	// If you call String on a reflect.Value of other type, it's better to
	// print something than to panic. Useful in debugging.
	return "<" + v.typ.String() + " Value>"
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It panics if v's Kind is not Chan.
// If the receive cannot finish without blocking, x is the zero Value.
// The boolean ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v Value) TryRecv() (x Value, ok bool) {
	v.mustBe(Chan)
	v.mustBeExported()
	return v.recv(true)
}

// TrySend attempts to send x on the channel v but will not block.
// It panics if v's Kind is not Chan.
// It returns true if the value was sent, false otherwise.
// As in Go, x's value must be assignable to the channel's element type.
func (v Value) TrySend(x Value) bool {
	v.mustBe(Chan)
	v.mustBeExported()
	return v.send(x, true)
}

// Type returns v's type.
func (v Value) Type() Type {
	f := v.flag
	if f == 0 {
		panic(&ValueError{"reflect.Value.Type", Invalid})
	}
	if f&flagMethod == 0 {
		// Easy case
		return v.typ
	}

	// Method value.
	// v.typ describes the receiver, not the method type.
	i := int(v.flag) >> flagMethodShift
	if v.typ.Kind() == Interface {
		// Method on interface.
		tt := (*interfaceType)(unsafe.Pointer(v.typ))
		if i < 0 || i >= len(tt.methods) {
			panic("reflect: internal error: invalid method index")
		}
		m := &tt.methods[i]
		return m.typ
	}
	// Method on concrete type.
	ut := v.typ.uncommon()
	if ut == nil || i < 0 || i >= len(ut.methods) {
		panic("reflect: internal error: invalid method index")
	}
	m := &ut.methods[i]
	return m.mtyp
}

// Uint returns v's underlying value, as a uint64.
// It panics if v's Kind is not Uint, Uintptr, Uint8, Uint16, Uint32, or Uint64.
func (v Value) Uint() uint64 {
	k := v.kind()
	var p unsafe.Pointer
	if v.flag&flagIndir != 0 {
		p = v.val
	} else {
		// The escape analysis is good enough that &v.val
		// does not trigger a heap allocation.
		p = unsafe.Pointer(&v.val)
	}
	switch k {
	case Uint:
		return uint64(*(*uint)(p))
	case Uint8:
		return uint64(*(*uint8)(p))
	case Uint16:
		return uint64(*(*uint16)(p))
	case Uint32:
		return uint64(*(*uint32)(p))
	case Uint64:
		return uint64(*(*uint64)(p))
	case Uintptr:
		return uint64(*(*uintptr)(p))
	}
	panic(&ValueError{"reflect.Value.Uint", k})
}

// UnsafeAddr returns a pointer to v's data.
// It is for advanced clients that also import the "unsafe" package.
// It panics if v is not addressable.
func (v Value) UnsafeAddr() uintptr {
	if v.typ == nil {
		panic(&ValueError{"reflect.Value.UnsafeAddr", Invalid})
	}
	if v.flag&flagAddr == 0 {
		panic("reflect.Value.UnsafeAddr of unaddressable value")
	}
	return uintptr(v.val)
}

// StringHeader is the runtime representation of a string.
// It cannot be used safely or portably and its representation may
// change in a later release.
// Moreover, the Data field is not sufficient to guarantee the data
// it references will not be garbage collected, so programs must keep
// a separate, correctly typed pointer to the underlying data.
type StringHeader struct {
	Data uintptr
	Len  int
}

// SliceHeader is the runtime representation of a slice.
// It cannot be used safely or portably and its representation may
// change in a later release.
// Moreover, the Data field is not sufficient to guarantee the data
// it references will not be garbage collected, so programs must keep
// a separate, correctly typed pointer to the underlying data.
type SliceHeader struct {
	Data uintptr
	Len  int
	Cap  int
}

func typesMustMatch(what string, t1, t2 Type) {
	if t1 != t2 {
		panic(what + ": " + t1.String() + " != " + t2.String())
	}
}

// grow grows the slice s so that it can hold extra more values, allocating
// more capacity if needed. It also returns the old and new slice lengths.
func grow(s Value, extra int) (Value, int, int) {
	i0 := s.Len()
	i1 := i0 + extra
	if i1 < i0 {
		panic("reflect.Append: slice overflow")
	}
	m := s.Cap()
	if i1 <= m {
		return s.Slice(0, i1), i0, i1
	}
	if m == 0 {
		m = extra
	} else {
		for m < i1 {
			if i0 < 1024 {
				m += m
			} else {
				m += m / 4
			}
		}
	}
	t := MakeSlice(s.Type(), i1, m)
	Copy(t, s)
	return t, i0, i1
}

// Append appends the values x to a slice s and returns the resulting slice.
// As in Go, each x's value must be assignable to the slice's element type.
func Append(s Value, x ...Value) Value {
	s.mustBe(Slice)
	s, i0, i1 := grow(s, len(x))
	for i, j := i0, 0; i < i1; i, j = i+1, j+1 {
		s.Index(i).Set(x[j])
	}
	return s
}

// AppendSlice appends a slice t to a slice s and returns the resulting slice.
// The slices s and t must have the same element type.
func AppendSlice(s, t Value) Value {
	s.mustBe(Slice)
	t.mustBe(Slice)
	typesMustMatch("reflect.AppendSlice", s.Type().Elem(), t.Type().Elem())
	s, i0, i1 := grow(s, t.Len())
	Copy(s.Slice(i0, i1), t)
	return s
}

// Copy copies the contents of src into dst until either
// dst has been filled or src has been exhausted.
// It returns the number of elements copied.
// Dst and src each must have kind Slice or Array, and
// dst and src must have the same element type.
func Copy(dst, src Value) int {
	dk := dst.kind()
	if dk != Array && dk != Slice {
		panic(&ValueError{"reflect.Copy", dk})
	}
	if dk == Array {
		dst.mustBeAssignable()
	}
	dst.mustBeExported()

	sk := src.kind()
	if sk != Array && sk != Slice {
		panic(&ValueError{"reflect.Copy", sk})
	}
	src.mustBeExported()

	de := dst.typ.Elem()
	se := src.typ.Elem()
	typesMustMatch("reflect.Copy", de, se)

	n := dst.Len()
	if sn := src.Len(); n > sn {
		n = sn
	}

	// If sk is an in-line array, cannot take its address.
	// Instead, copy element by element.
	if src.flag&flagIndir == 0 {
		for i := 0; i < n; i++ {
			dst.Index(i).Set(src.Index(i))
		}
		return n
	}

	// Copy via memmove.
	var da, sa unsafe.Pointer
	if dk == Array {
		da = dst.val
	} else {
		da = unsafe.Pointer((*SliceHeader)(dst.val).Data)
	}
	if sk == Array {
		sa = src.val
	} else {
		sa = unsafe.Pointer((*SliceHeader)(src.val).Data)
	}
	memmove(da, sa, uintptr(n)*de.Size())
	return n
}

// A runtimeSelect is a single case passed to rselect.
// This must match ../runtime/chan.c:/runtimeSelect
type runtimeSelect struct {
	dir uintptr // 0, SendDir, or RecvDir
	typ *rtype  // channel type
	ch  iword   // interface word for channel
	val iword   // interface word for value (for SendDir)
}

// rselect runs a select. It returns the index of the chosen case,
// and if the case was a receive, the interface word of the received
// value and the conventional OK bool to indicate whether the receive
// corresponds to a sent value.
func rselect([]runtimeSelect) (chosen int, recv iword, recvOK bool)

// A SelectDir describes the communication direction of a select case.
type SelectDir int

// NOTE: These values must match ../runtime/chan.c:/SelectDir.

const (
	_             SelectDir = iota
	SelectSend              // case Chan <- Send
	SelectRecv              // case <-Chan:
	SelectDefault           // default
)

// A SelectCase describes a single case in a select operation.
// The kind of case depends on Dir, the communication direction.
//
// If Dir is SelectDefault, the case represents a default case.
// Chan and Send must be zero Values.
//
// If Dir is SelectSend, the case represents a send operation.
// Normally Chan's underlying value must be a channel, and Send's underlying value must be
// assignable to the channel's element type. As a special case, if Chan is a zero Value,
// then the case is ignored, and the field Send will also be ignored and may be either zero
// or non-zero.
//
// If Dir is SelectRecv, the case represents a receive operation.
// Normally Chan's underlying value must be a channel and Send must be a zero Value.
// If Chan is a zero Value, then the case is ignored, but Send must still be a zero Value.
// When a receive operation is selected, the received Value is returned by Select.
//
type SelectCase struct {
	Dir  SelectDir // direction of case
	Chan Value     // channel to use (for send or receive)
	Send Value     // value to send (for send)
}

// Select executes a select operation described by the list of cases.
// Like the Go select statement, it blocks until at least one of the cases
// can proceed, makes a uniform pseudo-random choice,
// and then executes that case. It returns the index of the chosen case
// and, if that case was a receive operation, the value received and a
// boolean indicating whether the value corresponds to a send on the channel
// (as opposed to a zero value received because the channel is closed).
func Select(cases []SelectCase) (chosen int, recv Value, recvOK bool) {
	// NOTE: Do not trust that caller is not modifying cases data underfoot.
	// The range is safe because the caller cannot modify our copy of the len
	// and each iteration makes its own copy of the value c.
	runcases := make([]runtimeSelect, len(cases))
	haveDefault := false
	for i, c := range cases {
		rc := &runcases[i]
		rc.dir = uintptr(c.Dir)
		switch c.Dir {
		default:
			panic("reflect.Select: invalid Dir")

		case SelectDefault: // default
			if haveDefault {
				panic("reflect.Select: multiple default cases")
			}
			haveDefault = true
			if c.Chan.IsValid() {
				panic("reflect.Select: default case has Chan value")
			}
			if c.Send.IsValid() {
				panic("reflect.Select: default case has Send value")
			}

		case SelectSend:
			ch := c.Chan
			if !ch.IsValid() {
				break
			}
			ch.mustBe(Chan)
			ch.mustBeExported()
			tt := (*chanType)(unsafe.Pointer(ch.typ))
			if ChanDir(tt.dir)&SendDir == 0 {
				panic("reflect.Select: SendDir case using recv-only channel")
			}
			rc.ch = ch.iword()
			rc.typ = &tt.rtype
			v := c.Send
			if !v.IsValid() {
				panic("reflect.Select: SendDir case missing Send value")
			}
			v.mustBeExported()
			v = v.assignTo("reflect.Select", tt.elem, nil)
			rc.val = v.iword()

		case SelectRecv:
			if c.Send.IsValid() {
				panic("reflect.Select: RecvDir case has Send value")
			}
			ch := c.Chan
			if !ch.IsValid() {
				break
			}
			ch.mustBe(Chan)
			ch.mustBeExported()
			tt := (*chanType)(unsafe.Pointer(ch.typ))
			rc.typ = &tt.rtype
			if ChanDir(tt.dir)&RecvDir == 0 {
				panic("reflect.Select: RecvDir case using send-only channel")
			}
			rc.ch = ch.iword()
		}
	}

	chosen, word, recvOK := rselect(runcases)
	if runcases[chosen].dir == uintptr(SelectRecv) {
		tt := (*chanType)(unsafe.Pointer(runcases[chosen].typ))
		typ := tt.elem
		fl := flag(typ.Kind()) << flagKindShift
		if typ.size > ptrSize {
			fl |= flagIndir
		}
		recv = Value{typ, unsafe.Pointer(word), fl}
	}
	return chosen, recv, recvOK
}

/*
 * constructors
 */

// implemented in package runtime
func unsafe_New(*rtype) unsafe.Pointer
func unsafe_NewArray(*rtype, int) unsafe.Pointer

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ Type, len, cap int) Value {
	if typ.Kind() != Slice {
		panic("reflect.MakeSlice of non-slice type")
	}
	if len < 0 {
		panic("reflect.MakeSlice: negative len")
	}
	if cap < 0 {
		panic("reflect.MakeSlice: negative cap")
	}
	if len > cap {
		panic("reflect.MakeSlice: len > cap")
	}

	// Declare slice so that gc can see the base pointer in it.
	var x []unsafe.Pointer

	// Reinterpret as *SliceHeader to edit.
	s := (*SliceHeader)(unsafe.Pointer(&x))
	s.Data = uintptr(unsafe_NewArray(typ.Elem().(*rtype), cap))
	s.Len = len
	s.Cap = cap

	return Value{typ.common(), unsafe.Pointer(&x), flagIndir | flag(Slice)<<flagKindShift}
}

// MakeChan creates a new channel with the specified type and buffer size.
func MakeChan(typ Type, buffer int) Value {
	if typ.Kind() != Chan {
		panic("reflect.MakeChan of non-chan type")
	}
	if buffer < 0 {
		panic("reflect.MakeChan: negative buffer size")
	}
	if typ.ChanDir() != BothDir {
		panic("reflect.MakeChan: unidirectional channel type")
	}
	ch := makechan(typ.(*rtype), uint64(buffer))
	return Value{typ.common(), unsafe.Pointer(ch), flag(Chan) << flagKindShift}
}

// MakeMap creates a new map of the specified type.
func MakeMap(typ Type) Value {
	if typ.Kind() != Map {
		panic("reflect.MakeMap of non-map type")
	}
	m := makemap(typ.(*rtype))
	return Value{typ.common(), unsafe.Pointer(m), flag(Map) << flagKindShift}
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a zero Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if v.Kind() != Ptr {
		return v
	}
	return v.Elem()
}

// ValueOf returns a new Value initialized to the concrete value
// stored in the interface i.  ValueOf(nil) returns the zero Value.
func ValueOf(i interface{}) Value {
	if i == nil {
		return Value{}
	}

	// TODO(rsc): Eliminate this terrible hack.
	// In the call to packValue, eface.typ doesn't escape,
	// and eface.word is an integer.  So it looks like
	// i (= eface) doesn't escape.  But really it does,
	// because eface.word is actually a pointer.
	escapes(i)

	// For an interface value with the noAddr bit set,
	// the representation is identical to an empty interface.
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	typ := eface.typ
	fl := flag(typ.Kind()) << flagKindShift
	if typ.size > ptrSize {
		fl |= flagIndir
	}
	return Value{typ, unsafe.Pointer(eface.word), fl}
}

// Zero returns a Value representing the zero value for the specified type.
// The result is different from the zero value of the Value struct,
// which represents no value at all.
// For example, Zero(TypeOf(42)) returns a Value with Kind Int and value 0.
// The returned value is neither addressable nor settable.
func Zero(typ Type) Value {
	if typ == nil {
		panic("reflect: Zero(nil)")
	}
	t := typ.common()
	fl := flag(t.Kind()) << flagKindShift
	if t.size <= ptrSize {
		return Value{t, nil, fl}
	}
	return Value{t, unsafe_New(typ.(*rtype)), fl | flagIndir}
}

// New returns a Value representing a pointer to a new zero value
// for the specified type.  That is, the returned Value's Type is PtrTo(t).
func New(typ Type) Value {
	if typ == nil {
		panic("reflect: New(nil)")
	}
	ptr := unsafe_New(typ.(*rtype))
	fl := flag(Ptr) << flagKindShift
	return Value{typ.common().ptrTo(), ptr, fl}
}

// NewAt returns a Value representing a pointer to a value of the
// specified type, using p as that pointer.
func NewAt(typ Type, p unsafe.Pointer) Value {
	fl := flag(Ptr) << flagKindShift
	return Value{typ.common().ptrTo(), p, fl}
}

// assignTo returns a value v that can be assigned directly to typ.
// It panics if v is not assignable to typ.
// For a conversion to an interface type, target is a suggested scratch space to use.
func (v Value) assignTo(context string, dst *rtype, target *interface{}) Value {
	if v.flag&flagMethod != 0 {
		v = makeMethodValue(context, v)
	}

	switch {
	case directlyAssignable(dst, v.typ):
		// Overwrite type so that they match.
		// Same memory layout, so no harm done.
		v.typ = dst
		fl := v.flag & (flagRO | flagAddr | flagIndir)
		fl |= flag(dst.Kind()) << flagKindShift
		return Value{dst, v.val, fl}

	case implements(dst, v.typ):
		if target == nil {
			target = new(interface{})
		}
		x := valueInterface(v, false)
		if dst.NumMethod() == 0 {
			*target = x
		} else {
			ifaceE2I(dst, x, unsafe.Pointer(target))
		}
		return Value{dst, unsafe.Pointer(target), flagIndir | flag(Interface)<<flagKindShift}
	}

	// Failed.
	panic(context + ": value of type " + v.typ.String() + " is not assignable to type " + dst.String())
}

// Convert returns the value v converted to type t.
// If the usual Go conversion rules do not allow conversion
// of the value v to type t, Convert panics.
func (v Value) Convert(t Type) Value {
	if v.flag&flagMethod != 0 {
		v = makeMethodValue("Convert", v)
	}
	op := convertOp(t.common(), v.typ)
	if op == nil {
		panic("reflect.Value.Convert: value of type " + v.typ.String() + " cannot be converted to type " + t.String())
	}
	return op(v, t)
}

// convertOp returns the function to convert a value of type src
// to a value of type dst. If the conversion is illegal, convertOp returns nil.
func convertOp(dst, src *rtype) func(Value, Type) Value {
	switch src.Kind() {
	case Int, Int8, Int16, Int32, Int64:
		switch dst.Kind() {
		case Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
			return cvtInt
		case Float32, Float64:
			return cvtIntFloat
		case String:
			return cvtIntString
		}

	case Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
		switch dst.Kind() {
		case Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
			return cvtUint
		case Float32, Float64:
			return cvtUintFloat
		case String:
			return cvtUintString
		}

	case Float32, Float64:
		switch dst.Kind() {
		case Int, Int8, Int16, Int32, Int64:
			return cvtFloatInt
		case Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
			return cvtFloatUint
		case Float32, Float64:
			return cvtFloat
		}

	case Complex64, Complex128:
		switch dst.Kind() {
		case Complex64, Complex128:
			return cvtComplex
		}

	case String:
		if dst.Kind() == Slice && dst.Elem().PkgPath() == "" {
			switch dst.Elem().Kind() {
			case Uint8:
				return cvtStringBytes
			case Int32:
				return cvtStringRunes
			}
		}

	case Slice:
		if dst.Kind() == String && src.Elem().PkgPath() == "" {
			switch src.Elem().Kind() {
			case Uint8:
				return cvtBytesString
			case Int32:
				return cvtRunesString
			}
		}
	}

	// dst and src have same underlying type.
	if haveIdenticalUnderlyingType(dst, src) {
		return cvtDirect
	}

	// dst and src are unnamed pointer types with same underlying base type.
	if dst.Kind() == Ptr && dst.Name() == "" &&
		src.Kind() == Ptr && src.Name() == "" &&
		haveIdenticalUnderlyingType(dst.Elem().common(), src.Elem().common()) {
		return cvtDirect
	}

	if implements(dst, src) {
		if src.Kind() == Interface {
			return cvtI2I
		}
		return cvtT2I
	}

	return nil
}

// makeInt returns a Value of type t equal to bits (possibly truncated),
// where t is a signed or unsigned int type.
func makeInt(f flag, bits uint64, t Type) Value {
	typ := t.common()
	if typ.size > ptrSize {
		// Assume ptrSize >= 4, so this must be uint64.
		ptr := unsafe_New(typ)
		*(*uint64)(unsafe.Pointer(ptr)) = bits
		return Value{typ, ptr, f | flag(typ.Kind())<<flagKindShift}
	}
	var w iword
	switch typ.size {
	case 1:
		*(*uint8)(unsafe.Pointer(&w)) = uint8(bits)
	case 2:
		*(*uint16)(unsafe.Pointer(&w)) = uint16(bits)
	case 4:
		*(*uint32)(unsafe.Pointer(&w)) = uint32(bits)
	case 8:
		*(*uint64)(unsafe.Pointer(&w)) = uint64(bits)
	}
	return Value{typ, unsafe.Pointer(w), f | flag(typ.Kind())<<flagKindShift}
}

// makeFloat returns a Value of type t equal to v (possibly truncated to float32),
// where t is a float32 or float64 type.
func makeFloat(f flag, v float64, t Type) Value {
	typ := t.common()
	if typ.size > ptrSize {
		// Assume ptrSize >= 4, so this must be float64.
		ptr := unsafe_New(typ)
		*(*float64)(unsafe.Pointer(ptr)) = v
		return Value{typ, ptr, f | flag(typ.Kind())<<flagKindShift}
	}

	var w iword
	switch typ.size {
	case 4:
		*(*float32)(unsafe.Pointer(&w)) = float32(v)
	case 8:
		*(*float64)(unsafe.Pointer(&w)) = v
	}
	return Value{typ, unsafe.Pointer(w), f | flag(typ.Kind())<<flagKindShift}
}

// makeComplex returns a Value of type t equal to v (possibly truncated to complex64),
// where t is a complex64 or complex128 type.
func makeComplex(f flag, v complex128, t Type) Value {
	typ := t.common()
	if typ.size > ptrSize {
		ptr := unsafe_New(typ)
		switch typ.size {
		case 8:
			*(*complex64)(unsafe.Pointer(ptr)) = complex64(v)
		case 16:
			*(*complex128)(unsafe.Pointer(ptr)) = v
		}
		return Value{typ, ptr, f | flag(typ.Kind())<<flagKindShift}
	}

	// Assume ptrSize <= 8 so this must be complex64.
	var w iword
	*(*complex64)(unsafe.Pointer(&w)) = complex64(v)
	return Value{typ, unsafe.Pointer(w), f | flag(typ.Kind())<<flagKindShift}
}

func makeString(f flag, v string, t Type) Value {
	ret := New(t).Elem()
	ret.SetString(v)
	ret.flag = ret.flag&^flagAddr | f
	return ret
}

func makeBytes(f flag, v []byte, t Type) Value {
	ret := New(t).Elem()
	ret.SetBytes(v)
	ret.flag = ret.flag&^flagAddr | f
	return ret
}

func makeRunes(f flag, v []rune, t Type) Value {
	ret := New(t).Elem()
	ret.setRunes(v)
	ret.flag = ret.flag&^flagAddr | f
	return ret
}

// These conversion functions are returned by convertOp
// for classes of conversions. For example, the first function, cvtInt,
// takes any value v of signed int type and returns the value converted
// to type t, where t is any signed or unsigned int type.

// convertOp: intXX -> [u]intXX
func cvtInt(v Value, t Type) Value {
	return makeInt(v.flag&flagRO, uint64(v.Int()), t)
}

// convertOp: uintXX -> [u]intXX
func cvtUint(v Value, t Type) Value {
	return makeInt(v.flag&flagRO, v.Uint(), t)
}

// convertOp: floatXX -> intXX
func cvtFloatInt(v Value, t Type) Value {
	return makeInt(v.flag&flagRO, uint64(int64(v.Float())), t)
}

// convertOp: floatXX -> uintXX
func cvtFloatUint(v Value, t Type) Value {
	return makeInt(v.flag&flagRO, uint64(v.Float()), t)
}

// convertOp: intXX -> floatXX
func cvtIntFloat(v Value, t Type) Value {
	return makeFloat(v.flag&flagRO, float64(v.Int()), t)
}

// convertOp: uintXX -> floatXX
func cvtUintFloat(v Value, t Type) Value {
	return makeFloat(v.flag&flagRO, float64(v.Uint()), t)
}

// convertOp: floatXX -> floatXX
func cvtFloat(v Value, t Type) Value {
	return makeFloat(v.flag&flagRO, v.Float(), t)
}

// convertOp: complexXX -> complexXX
func cvtComplex(v Value, t Type) Value {
	return makeComplex(v.flag&flagRO, v.Complex(), t)
}

// convertOp: intXX -> string
func cvtIntString(v Value, t Type) Value {
	return makeString(v.flag&flagRO, string(v.Int()), t)
}

// convertOp: uintXX -> string
func cvtUintString(v Value, t Type) Value {
	return makeString(v.flag&flagRO, string(v.Uint()), t)
}

// convertOp: []byte -> string
func cvtBytesString(v Value, t Type) Value {
	return makeString(v.flag&flagRO, string(v.Bytes()), t)
}

// convertOp: string -> []byte
func cvtStringBytes(v Value, t Type) Value {
	return makeBytes(v.flag&flagRO, []byte(v.String()), t)
}

// convertOp: []rune -> string
func cvtRunesString(v Value, t Type) Value {
	return makeString(v.flag&flagRO, string(v.runes()), t)
}

// convertOp: string -> []rune
func cvtStringRunes(v Value, t Type) Value {
	return makeRunes(v.flag&flagRO, []rune(v.String()), t)
}

// convertOp: direct copy
func cvtDirect(v Value, typ Type) Value {
	f := v.flag
	t := typ.common()
	val := v.val
	if f&flagAddr != 0 {
		// indirect, mutable word - make a copy
		ptr := unsafe_New(t)
		memmove(ptr, val, t.size)
		val = ptr
		f &^= flagAddr
	}
	return Value{t, val, v.flag&flagRO | f}
}

// convertOp: concrete -> interface
func cvtT2I(v Value, typ Type) Value {
	target := new(interface{})
	x := valueInterface(v, false)
	if typ.NumMethod() == 0 {
		*target = x
	} else {
		ifaceE2I(typ.(*rtype), x, unsafe.Pointer(target))
	}
	return Value{typ.common(), unsafe.Pointer(target), v.flag&flagRO | flagIndir | flag(Interface)<<flagKindShift}
}

// convertOp: interface -> interface
func cvtI2I(v Value, typ Type) Value {
	if v.IsNil() {
		ret := Zero(typ)
		ret.flag |= v.flag & flagRO
		return ret
	}
	return cvtT2I(v.Elem(), typ)
}

// implemented in ../pkg/runtime
func chancap(ch iword) int
func chanclose(ch iword)
func chanlen(ch iword) int
func chanrecv(t *rtype, ch iword, nb bool) (val iword, selected, received bool)
func chansend(t *rtype, ch iword, val iword, nb bool) bool

func makechan(typ *rtype, size uint64) (ch iword)
func makemap(t *rtype) (m iword)
func mapaccess(t *rtype, m iword, key iword) (val iword, ok bool)
func mapassign(t *rtype, m iword, key, val iword, ok bool)
func mapiterinit(t *rtype, m iword) *byte
func mapiterkey(it *byte) (key iword, ok bool)
func mapiternext(it *byte)
func maplen(m iword) int

func call(fn, arg unsafe.Pointer, n uint32)
func ifaceE2I(t *rtype, src interface{}, dst unsafe.Pointer)

// Dummy annotation marking that the value x escapes,
// for use in cases where the reflect code is so clever that
// the compiler cannot follow.
func escapes(x interface{}) {
	if dummy.b {
		dummy.x = x
	}
}

var dummy struct {
	b bool
	x interface{}
}
