// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/bytealg"

// The Error interface identifies a run time error.
type Error interface {
	error

	// RuntimeError is a no-op function but
	// serves to distinguish types that are run time
	// errors from ordinary errors: a type is a
	// run time error if it has a RuntimeError method.
	RuntimeError()
}

// A TypeAssertionError explains a failed type assertion.
type TypeAssertionError struct {
	_interface    *_type
	concrete      *_type
	asserted      *_type
	missingMethod string // one method needed by Interface, missing from Concrete
}

func (*TypeAssertionError) RuntimeError() {}

func (e *TypeAssertionError) Error() string {
	inter := "interface"
	if e._interface != nil {
		inter = e._interface.string()
	}
	as := e.asserted.string()
	if e.concrete == nil {
		return "interface conversion: " + inter + " is nil, not " + as
	}
	cs := e.concrete.string()
	if e.missingMethod == "" {
		msg := "interface conversion: " + inter + " is " + cs + ", not " + as
		if cs == as {
			// provide slightly clearer error message
			if e.concrete.pkgpath() != e.asserted.pkgpath() {
				msg += " (types from different packages)"
			} else {
				msg += " (types from different scopes)"
			}
		}
		return msg
	}
	return "interface conversion: " + cs + " is not " + as +
		": missing method " + e.missingMethod
}

//go:nosplit
// itoa converts val to a decimal representation. The result is
// written somewhere within buf and the location of the result is returned.
// buf must be at least 20 bytes.
func itoa(buf []byte, val uint64) []byte {
	i := len(buf) - 1
	for val >= 10 {
		buf[i] = byte(val%10 + '0')
		i--
		val /= 10
	}
	buf[i] = byte(val + '0')
	return buf[i:]
}

// An errorString represents a runtime error described by a single string.
type errorString string

func (e errorString) RuntimeError() {}

func (e errorString) Error() string {
	return "runtime error: " + string(e)
}

type errorAddressString struct {
	msg  string  // error message
	addr uintptr // memory address where the error occurred
}

func (e errorAddressString) RuntimeError() {}

func (e errorAddressString) Error() string {
	return "runtime error: " + e.msg
}

// Addr returns the memory address where a fault occurred.
// The address provided is best-effort.
// The veracity of the result may depend on the platform.
// Errors providing this method will only be returned as
// a result of using runtime/debug.SetPanicOnFault.
func (e errorAddressString) Addr() uintptr {
	return e.addr
}

// plainError represents a runtime error described a string without
// the prefix "runtime error: " after invoking errorString.Error().
// See Issue #14965.
type plainError string

func (e plainError) RuntimeError() {}

func (e plainError) Error() string {
	return string(e)
}

// A boundsError represents an indexing or slicing operation gone wrong.
type boundsError struct {
	x int64
	y int
	// Values in an index or slice expression can be signed or unsigned.
	// That means we'd need 65 bits to encode all possible indexes, from -2^63 to 2^64-1.
	// Instead, we keep track of whether x should be interpreted as signed or unsigned.
	// y is known to be nonnegative and to fit in an int.
	signed bool
	code   boundsErrorCode
}

type boundsErrorCode uint8

const (
	boundsIndex boundsErrorCode = iota // s[x], 0 <= x < len(s) failed

	boundsSliceAlen // s[?:x], 0 <= x <= len(s) failed
	boundsSliceAcap // s[?:x], 0 <= x <= cap(s) failed
	boundsSliceB    // s[x:y], 0 <= x <= y failed (but boundsSliceA didn't happen)

	boundsSlice3Alen // s[?:?:x], 0 <= x <= len(s) failed
	boundsSlice3Acap // s[?:?:x], 0 <= x <= cap(s) failed
	boundsSlice3B    // s[?:x:y], 0 <= x <= y failed (but boundsSlice3A didn't happen)
	boundsSlice3C    // s[x:y:?], 0 <= x <= y failed (but boundsSlice3A/B didn't happen)

	boundsConvert // (*[x]T)(s), 0 <= x <= len(s) failed
	// Note: in the above, len(s) and cap(s) are stored in y
)

// boundsErrorFmts provide error text for various out-of-bounds panics.
// Note: if you change these strings, you should adjust the size of the buffer
// in boundsError.Error below as well.
var boundsErrorFmts = [...]string{
	boundsIndex:      "index out of range [%x] with length %y",
	boundsSliceAlen:  "slice bounds out of range [:%x] with length %y",
	boundsSliceAcap:  "slice bounds out of range [:%x] with capacity %y",
	boundsSliceB:     "slice bounds out of range [%x:%y]",
	boundsSlice3Alen: "slice bounds out of range [::%x] with length %y",
	boundsSlice3Acap: "slice bounds out of range [::%x] with capacity %y",
	boundsSlice3B:    "slice bounds out of range [:%x:%y]",
	boundsSlice3C:    "slice bounds out of range [%x:%y:]",
	boundsConvert:    "cannot convert slice with length %y to pointer to array with length %x",
}

// boundsNegErrorFmts are overriding formats if x is negative. In this case there's no need to report y.
var boundsNegErrorFmts = [...]string{
	boundsIndex:      "index out of range [%x]",
	boundsSliceAlen:  "slice bounds out of range [:%x]",
	boundsSliceAcap:  "slice bounds out of range [:%x]",
	boundsSliceB:     "slice bounds out of range [%x:]",
	boundsSlice3Alen: "slice bounds out of range [::%x]",
	boundsSlice3Acap: "slice bounds out of range [::%x]",
	boundsSlice3B:    "slice bounds out of range [:%x:]",
	boundsSlice3C:    "slice bounds out of range [%x::]",
}

func (e boundsError) RuntimeError() {}

func appendIntStr(b []byte, v int64, signed bool) []byte {
	if signed && v < 0 {
		b = append(b, '-')
		v = -v
	}
	var buf [20]byte
	b = append(b, itoa(buf[:], uint64(v))...)
	return b
}

func (e boundsError) Error() string {
	fmt := boundsErrorFmts[e.code]
	if e.signed && e.x < 0 {
		fmt = boundsNegErrorFmts[e.code]
	}
	// max message length is 99: "runtime error: slice bounds out of range [::%x] with capacity %y"
	// x can be at most 20 characters. y can be at most 19.
	b := make([]byte, 0, 100)
	b = append(b, "runtime error: "...)
	for i := 0; i < len(fmt); i++ {
		c := fmt[i]
		if c != '%' {
			b = append(b, c)
			continue
		}
		i++
		switch fmt[i] {
		case 'x':
			b = appendIntStr(b, e.x, e.signed)
		case 'y':
			b = appendIntStr(b, int64(e.y), true)
		}
	}
	return string(b)
}

type stringer interface {
	String() string
}

// printany prints an argument passed to panic.
// If panic is called with a value that has a String or Error method,
// it has already been converted into a string by preprintpanics.
func printany(i interface{}) {
	switch v := i.(type) {
	case nil:
		print("nil")
	case bool:
		print(v)
	case int:
		print(v)
	case int8:
		print(v)
	case int16:
		print(v)
	case int32:
		print(v)
	case int64:
		print(v)
	case uint:
		print(v)
	case uint8:
		print(v)
	case uint16:
		print(v)
	case uint32:
		print(v)
	case uint64:
		print(v)
	case uintptr:
		print(v)
	case float32:
		print(v)
	case float64:
		print(v)
	case complex64:
		print(v)
	case complex128:
		print(v)
	case string:
		print(v)
	default:
		printanycustomtype(i)
	}
}

func printanycustomtype(i interface{}) {
	eface := efaceOf(&i)
	typestring := eface._type.string()

	switch eface._type.kind {
	case kindString:
		print(typestring, `("`, *(*string)(eface.data), `")`)
	case kindBool:
		print(typestring, "(", *(*bool)(eface.data), ")")
	case kindInt:
		print(typestring, "(", *(*int)(eface.data), ")")
	case kindInt8:
		print(typestring, "(", *(*int8)(eface.data), ")")
	case kindInt16:
		print(typestring, "(", *(*int16)(eface.data), ")")
	case kindInt32:
		print(typestring, "(", *(*int32)(eface.data), ")")
	case kindInt64:
		print(typestring, "(", *(*int64)(eface.data), ")")
	case kindUint:
		print(typestring, "(", *(*uint)(eface.data), ")")
	case kindUint8:
		print(typestring, "(", *(*uint8)(eface.data), ")")
	case kindUint16:
		print(typestring, "(", *(*uint16)(eface.data), ")")
	case kindUint32:
		print(typestring, "(", *(*uint32)(eface.data), ")")
	case kindUint64:
		print(typestring, "(", *(*uint64)(eface.data), ")")
	case kindUintptr:
		print(typestring, "(", *(*uintptr)(eface.data), ")")
	case kindFloat32:
		print(typestring, "(", *(*float32)(eface.data), ")")
	case kindFloat64:
		print(typestring, "(", *(*float64)(eface.data), ")")
	case kindComplex64:
		print(typestring, *(*complex64)(eface.data))
	case kindComplex128:
		print(typestring, *(*complex128)(eface.data))
	default:
		print("(", typestring, ") ", eface.data)
	}
}

// panicwrap generates a panic for a call to a wrapped value method
// with a nil pointer receiver.
//
// It is called from the generated wrapper code.
func panicwrap() {
	pc := getcallerpc()
	name := funcname(findfunc(pc))
	// name is something like "main.(*T).F".
	// We want to extract pkg ("main"), typ ("T"), and meth ("F").
	// Do it by finding the parens.
	i := bytealg.IndexByteString(name, '(')
	if i < 0 {
		throw("panicwrap: no ( in " + name)
	}
	pkg := name[:i-1]
	if i+2 >= len(name) || name[i-1:i+2] != ".(*" {
		throw("panicwrap: unexpected string after package name: " + name)
	}
	name = name[i+2:]
	i = bytealg.IndexByteString(name, ')')
	if i < 0 {
		throw("panicwrap: no ) in " + name)
	}
	if i+2 >= len(name) || name[i:i+2] != ")." {
		throw("panicwrap: unexpected string after type name: " + name)
	}
	typ := name[:i]
	meth := name[i+2:]
	panic(plainError("value method " + pkg + "." + typ + "." + meth + " called using nil *" + typ + " pointer"))
}
