// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package interp

// Values
//
// All interpreter values are "boxed" in the empty interface, value.
// The range of possible dynamic types within value are:
//
// - bool
// - numbers (all built-in int/float/complex types are distinguished)
// - string
// - map[value]value --- maps for which  usesBuiltinMap(keyType)
//   *hashmap        --- maps for which !usesBuiltinMap(keyType)
// - chan value
// - []value --- slices
// - iface --- interfaces.
// - structure --- structs.  Fields are ordered and accessed by numeric indices.
// - array --- arrays.
// - *value --- pointers.  Careful: *value is a distinct type from *array etc.
// - *ssa.Function \
//   *ssa.Builtin   } --- functions.  A nil 'func' is always of type *ssa.Function.
//   *closure      /
// - tuple --- as returned by Return, Next, "value,ok" modes, etc.
// - iter --- iterators from 'range' over map or string.
// - bad --- a poison pill for locals that have gone out of scope.
// - rtype -- the interpreter's concrete implementation of reflect.Type
//
// Note that nil is not on this list.
//
// Pay close attention to whether or not the dynamic type is a pointer.
// The compiler cannot help you since value is an empty interface.

import (
	"bytes"
	"fmt"
	"go/types"
	"io"
	"reflect"
	"strings"
	"sync"
	"unsafe"

	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/go/types/typeutil"
)

type value interface{}

type tuple []value

type array []value

type iface struct {
	t types.Type // never an "untyped" type
	v value
}

type structure []value

// For map, array, *array, slice, string or channel.
type iter interface {
	// next returns a Tuple (key, value, ok).
	// key and value are unaliased, e.g. copies of the sequence element.
	next() tuple
}

type closure struct {
	Fn  *ssa.Function
	Env []value
}

type bad struct{}

type rtype struct {
	t types.Type
}

// Hash functions and equivalence relation:

// hashString computes the FNV hash of s.
func hashString(s string) int {
	var h uint32
	for i := 0; i < len(s); i++ {
		h ^= uint32(s[i])
		h *= 16777619
	}
	return int(h)
}

var (
	mu     sync.Mutex
	hasher = typeutil.MakeHasher()
)

// hashType returns a hash for t such that
// types.Identical(x, y) => hashType(x) == hashType(y).
func hashType(t types.Type) int {
	mu.Lock()
	h := int(hasher.Hash(t))
	mu.Unlock()
	return h
}

// usesBuiltinMap returns true if the built-in hash function and
// equivalence relation for type t are consistent with those of the
// interpreter's representation of type t.  Such types are: all basic
// types (bool, numbers, string), pointers and channels.
//
// usesBuiltinMap returns false for types that require a custom map
// implementation: interfaces, arrays and structs.
//
// Panic ensues if t is an invalid map key type: function, map or slice.
func usesBuiltinMap(t types.Type) bool {
	switch t := t.(type) {
	case *types.Basic, *types.Chan, *types.Pointer:
		return true
	case *types.Named:
		return usesBuiltinMap(t.Underlying())
	case *types.Interface, *types.Array, *types.Struct:
		return false
	}
	panic(fmt.Sprintf("invalid map key type: %T", t))
}

func (x array) eq(t types.Type, _y interface{}) bool {
	y := _y.(array)
	tElt := t.Underlying().(*types.Array).Elem()
	for i, xi := range x {
		if !equals(tElt, xi, y[i]) {
			return false
		}
	}
	return true
}

func (x array) hash(t types.Type) int {
	h := 0
	tElt := t.Underlying().(*types.Array).Elem()
	for _, xi := range x {
		h += hash(tElt, xi)
	}
	return h
}

func (x structure) eq(t types.Type, _y interface{}) bool {
	y := _y.(structure)
	tStruct := t.Underlying().(*types.Struct)
	for i, n := 0, tStruct.NumFields(); i < n; i++ {
		if f := tStruct.Field(i); !f.Anonymous() {
			if !equals(f.Type(), x[i], y[i]) {
				return false
			}
		}
	}
	return true
}

func (x structure) hash(t types.Type) int {
	tStruct := t.Underlying().(*types.Struct)
	h := 0
	for i, n := 0, tStruct.NumFields(); i < n; i++ {
		if f := tStruct.Field(i); !f.Anonymous() {
			h += hash(f.Type(), x[i])
		}
	}
	return h
}

// nil-tolerant variant of types.Identical.
func sameType(x, y types.Type) bool {
	if x == nil {
		return y == nil
	}
	return y != nil && types.Identical(x, y)
}

func (x iface) eq(t types.Type, _y interface{}) bool {
	y := _y.(iface)
	return sameType(x.t, y.t) && (x.t == nil || equals(x.t, x.v, y.v))
}

func (x iface) hash(_ types.Type) int {
	return hashType(x.t)*8581 + hash(x.t, x.v)
}

func (x rtype) hash(_ types.Type) int {
	return hashType(x.t)
}

func (x rtype) eq(_ types.Type, y interface{}) bool {
	return types.Identical(x.t, y.(rtype).t)
}

// equals returns true iff x and y are equal according to Go's
// linguistic equivalence relation for type t.
// In a well-typed program, the dynamic types of x and y are
// guaranteed equal.
func equals(t types.Type, x, y value) bool {
	switch x := x.(type) {
	case bool:
		return x == y.(bool)
	case int:
		return x == y.(int)
	case int8:
		return x == y.(int8)
	case int16:
		return x == y.(int16)
	case int32:
		return x == y.(int32)
	case int64:
		return x == y.(int64)
	case uint:
		return x == y.(uint)
	case uint8:
		return x == y.(uint8)
	case uint16:
		return x == y.(uint16)
	case uint32:
		return x == y.(uint32)
	case uint64:
		return x == y.(uint64)
	case uintptr:
		return x == y.(uintptr)
	case float32:
		return x == y.(float32)
	case float64:
		return x == y.(float64)
	case complex64:
		return x == y.(complex64)
	case complex128:
		return x == y.(complex128)
	case string:
		return x == y.(string)
	case *value:
		return x == y.(*value)
	case chan value:
		return x == y.(chan value)
	case structure:
		return x.eq(t, y)
	case array:
		return x.eq(t, y)
	case iface:
		return x.eq(t, y)
	case rtype:
		return x.eq(t, y)
	}

	// Since map, func and slice don't support comparison, this
	// case is only reachable if one of x or y is literally nil
	// (handled in eqnil) or via interface{} values.
	panic(fmt.Sprintf("comparing uncomparable type %s", t))
}

// Returns an integer hash of x such that equals(x, y) => hash(x) == hash(y).
func hash(t types.Type, x value) int {
	switch x := x.(type) {
	case bool:
		if x {
			return 1
		}
		return 0
	case int:
		return x
	case int8:
		return int(x)
	case int16:
		return int(x)
	case int32:
		return int(x)
	case int64:
		return int(x)
	case uint:
		return int(x)
	case uint8:
		return int(x)
	case uint16:
		return int(x)
	case uint32:
		return int(x)
	case uint64:
		return int(x)
	case uintptr:
		return int(x)
	case float32:
		return int(x)
	case float64:
		return int(x)
	case complex64:
		return int(real(x))
	case complex128:
		return int(real(x))
	case string:
		return hashString(x)
	case *value:
		return int(uintptr(unsafe.Pointer(x)))
	case chan value:
		return int(uintptr(reflect.ValueOf(x).Pointer()))
	case structure:
		return x.hash(t)
	case array:
		return x.hash(t)
	case iface:
		return x.hash(t)
	case rtype:
		return x.hash(t)
	}
	panic(fmt.Sprintf("%T is unhashable", x))
}

// reflect.Value struct values don't have a fixed shape, since the
// payload can be a scalar or an aggregate depending on the instance.
// So store (and load) can't simply use recursion over the shape of the
// rhs value, or the lhs, to copy the value; we need the static type
// information.  (We can't make reflect.Value a new basic data type
// because its "structness" is exposed to Go programs.)

// load returns the value of type T in *addr.
func load(T types.Type, addr *value) value {
	switch T := T.Underlying().(type) {
	case *types.Struct:
		v := (*addr).(structure)
		a := make(structure, len(v))
		for i := range a {
			a[i] = load(T.Field(i).Type(), &v[i])
		}
		return a
	case *types.Array:
		v := (*addr).(array)
		a := make(array, len(v))
		for i := range a {
			a[i] = load(T.Elem(), &v[i])
		}
		return a
	default:
		return *addr
	}
}

// store stores value v of type T into *addr.
func store(T types.Type, addr *value, v value) {
	switch T := T.Underlying().(type) {
	case *types.Struct:
		lhs := (*addr).(structure)
		rhs := v.(structure)
		for i := range lhs {
			store(T.Field(i).Type(), &lhs[i], rhs[i])
		}
	case *types.Array:
		lhs := (*addr).(array)
		rhs := v.(array)
		for i := range lhs {
			store(T.Elem(), &lhs[i], rhs[i])
		}
	default:
		*addr = v
	}
}

// Prints in the style of built-in println.
// (More or less; in gc println is actually a compiler intrinsic and
// can distinguish println(1) from println(interface{}(1)).)
func writeValue(buf *bytes.Buffer, v value) {
	switch v := v.(type) {
	case nil, bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr, float32, float64, complex64, complex128, string:
		fmt.Fprintf(buf, "%v", v)

	case map[value]value:
		buf.WriteString("map[")
		sep := ""
		for k, e := range v {
			buf.WriteString(sep)
			sep = " "
			writeValue(buf, k)
			buf.WriteString(":")
			writeValue(buf, e)
		}
		buf.WriteString("]")

	case *hashmap:
		buf.WriteString("map[")
		sep := " "
		for _, e := range v.table {
			for e != nil {
				buf.WriteString(sep)
				sep = " "
				writeValue(buf, e.key)
				buf.WriteString(":")
				writeValue(buf, e.value)
				e = e.next
			}
		}
		buf.WriteString("]")

	case chan value:
		fmt.Fprintf(buf, "%v", v) // (an address)

	case *value:
		if v == nil {
			buf.WriteString("<nil>")
		} else {
			fmt.Fprintf(buf, "%p", v)
		}

	case iface:
		fmt.Fprintf(buf, "(%s, ", v.t)
		writeValue(buf, v.v)
		buf.WriteString(")")

	case structure:
		buf.WriteString("{")
		for i, e := range v {
			if i > 0 {
				buf.WriteString(" ")
			}
			writeValue(buf, e)
		}
		buf.WriteString("}")

	case array:
		buf.WriteString("[")
		for i, e := range v {
			if i > 0 {
				buf.WriteString(" ")
			}
			writeValue(buf, e)
		}
		buf.WriteString("]")

	case []value:
		buf.WriteString("[")
		for i, e := range v {
			if i > 0 {
				buf.WriteString(" ")
			}
			writeValue(buf, e)
		}
		buf.WriteString("]")

	case *ssa.Function, *ssa.Builtin, *closure:
		fmt.Fprintf(buf, "%p", v) // (an address)

	case rtype:
		buf.WriteString(v.t.String())

	case tuple:
		// Unreachable in well-formed Go programs
		buf.WriteString("(")
		for i, e := range v {
			if i > 0 {
				buf.WriteString(", ")
			}
			writeValue(buf, e)
		}
		buf.WriteString(")")

	default:
		fmt.Fprintf(buf, "<%T>", v)
	}
}

// Implements printing of Go values in the style of built-in println.
func toString(v value) string {
	var b bytes.Buffer
	writeValue(&b, v)
	return b.String()
}

// ------------------------------------------------------------------------
// Iterators

type stringIter struct {
	*strings.Reader
	i int
}

func (it *stringIter) next() tuple {
	okv := make(tuple, 3)
	ch, n, err := it.ReadRune()
	ok := err != io.EOF
	okv[0] = ok
	if ok {
		okv[1] = it.i
		okv[2] = ch
	}
	it.i += n
	return okv
}

type mapIter chan [2]value

func (it mapIter) next() tuple {
	kv, ok := <-it
	return tuple{ok, kv[0], kv[1]}
}
