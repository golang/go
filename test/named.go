// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that basic operations on named types are valid
// and preserve the type.

package main

type Array [10]byte
type Bool bool
type Chan chan int
type Float float32
type Int int
type Map map[int]byte
type Slice []byte
type String string

// Calling these functions checks at compile time that the argument
// can be converted implicitly to (used as) the given type.
func asArray(Array)   {}
func asBool(Bool)     {}
func asChan(Chan)     {}
func asFloat(Float)   {}
func asInt(Int)       {}
func asMap(Map)       {}
func asSlice(Slice)   {}
func asString(String) {}

func (Map) M() {}


// These functions check at run time that the default type
// (in the absence of any implicit conversion hints)
// is the given type.
func isArray(x interface{})  { _ = x.(Array) }
func isBool(x interface{})   { _ = x.(Bool) }
func isChan(x interface{})   { _ = x.(Chan) }
func isFloat(x interface{})  { _ = x.(Float) }
func isInt(x interface{})    { _ = x.(Int) }
func isMap(x interface{})    { _ = x.(Map) }
func isSlice(x interface{})  { _ = x.(Slice) }
func isString(x interface{}) { _ = x.(String) }

func main() {
	var (
		a     Array
		b     Bool   = true
		c     Chan   = make(Chan)
		f     Float  = 1
		i     Int    = 1
		m     Map    = make(Map)
		slice Slice  = make(Slice, 10)
		str   String = "hello"
	)

	asArray(a)
	isArray(a)
	asArray(*&a)
	isArray(*&a)
	asArray(Array{})
	isArray(Array{})

	asBool(b)
	isBool(b)
	asBool(!b)
	isBool(!b)
	asBool(true)
	asBool(*&b)
	isBool(*&b)
	asBool(Bool(true))
	isBool(Bool(true))

	asChan(c)
	isChan(c)
	asChan(make(Chan))
	isChan(make(Chan))
	asChan(*&c)
	isChan(*&c)
	asChan(Chan(nil))
	isChan(Chan(nil))

	asFloat(f)
	isFloat(f)
	asFloat(-f)
	isFloat(-f)
	asFloat(+f)
	isFloat(+f)
	asFloat(f + 1)
	isFloat(f + 1)
	asFloat(1 + f)
	isFloat(1 + f)
	asFloat(f + f)
	isFloat(f + f)
	f++
	f += 2
	asFloat(f - 1)
	isFloat(f - 1)
	asFloat(1 - f)
	isFloat(1 - f)
	asFloat(f - f)
	isFloat(f - f)
	f--
	f -= 2
	asFloat(f * 2.5)
	isFloat(f * 2.5)
	asFloat(2.5 * f)
	isFloat(2.5 * f)
	asFloat(f * f)
	isFloat(f * f)
	f *= 4
	asFloat(f / 2.5)
	isFloat(f / 2.5)
	asFloat(2.5 / f)
	isFloat(2.5 / f)
	asFloat(f / f)
	isFloat(f / f)
	f /= 4
	asFloat(f)
	isFloat(f)
	f = 5
	asFloat(*&f)
	isFloat(*&f)
	asFloat(234)
	asFloat(Float(234))
	isFloat(Float(234))
	asFloat(1.2)
	asFloat(Float(i))
	isFloat(Float(i))

	asInt(i)
	isInt(i)
	asInt(-i)
	isInt(-i)
	asInt(^i)
	isInt(^i)
	asInt(+i)
	isInt(+i)
	asInt(i + 1)
	isInt(i + 1)
	asInt(1 + i)
	isInt(1 + i)
	asInt(i + i)
	isInt(i + i)
	i++
	i += 1
	asInt(i - 1)
	isInt(i - 1)
	asInt(1 - i)
	isInt(1 - i)
	asInt(i - i)
	isInt(i - i)
	i--
	i -= 1
	asInt(i * 2)
	isInt(i * 2)
	asInt(2 * i)
	isInt(2 * i)
	asInt(i * i)
	isInt(i * i)
	i *= 2
	asInt(i / 5)
	isInt(i / 5)
	asInt(5 / i)
	isInt(5 / i)
	asInt(i / i)
	isInt(i / i)
	i /= 2
	asInt(i % 5)
	isInt(i % 5)
	asInt(5 % i)
	isInt(5 % i)
	asInt(i % i)
	isInt(i % i)
	i %= 2
	asInt(i & 5)
	isInt(i & 5)
	asInt(5 & i)
	isInt(5 & i)
	asInt(i & i)
	isInt(i & i)
	i &= 2
	asInt(i &^ 5)
	isInt(i &^ 5)
	asInt(5 &^ i)
	isInt(5 &^ i)
	asInt(i &^ i)
	isInt(i &^ i)
	i &^= 2
	asInt(i | 5)
	isInt(i | 5)
	asInt(5 | i)
	isInt(5 | i)
	asInt(i | i)
	isInt(i | i)
	i |= 2
	asInt(i ^ 5)
	isInt(i ^ 5)
	asInt(5 ^ i)
	isInt(5 ^ i)
	asInt(i ^ i)
	isInt(i ^ i)
	i ^= 2
	asInt(i << 4)
	isInt(i << 4)
	i <<= 2
	asInt(i >> 4)
	isInt(i >> 4)
	i >>= 2
	asInt(i)
	isInt(i)
	asInt(0)
	asInt(Int(0))
	isInt(Int(0))
	i = 10
	asInt(*&i)
	isInt(*&i)
	asInt(23)
	asInt(Int(f))
	isInt(Int(f))

	asMap(m)
	isMap(m)
	asMap(nil)
	m = nil
	asMap(make(Map))
	isMap(make(Map))
	asMap(*&m)
	isMap(*&m)
	asMap(Map(nil))
	isMap(Map(nil))
	asMap(Map{})
	isMap(Map{})

	asSlice(slice)
	isSlice(slice)
	asSlice(make(Slice, 5))
	isSlice(make(Slice, 5))
	asSlice([]byte{1, 2, 3})
	asSlice([]byte{1, 2, 3}[0:2])
	asSlice(slice[0:4])
	isSlice(slice[0:4])
	asSlice(slice[3:8])
	isSlice(slice[3:8])
	asSlice(nil)
	asSlice(Slice(nil))
	isSlice(Slice(nil))
	slice = nil
	asSlice(Slice{1, 2, 3})
	isSlice(Slice{1, 2, 3})
	asSlice(Slice{})
	isSlice(Slice{})
	asSlice(*&slice)
	isSlice(*&slice)

	asString(str)
	isString(str)
	asString(str + "a")
	isString(str + "a")
	asString("a" + str)
	isString("a" + str)
	asString(str + str)
	isString(str + str)
	str += "a"
	str += str
	asString(String('a'))
	isString(String('a'))
	asString(String([]byte(slice)))
	isString(String([]byte(slice)))
	asString(String([]byte(nil)))
	isString(String([]byte(nil)))
	asString("hello")
	asString(String("hello"))
	isString(String("hello"))
	str = "hello"
	isString(str)
	asString(*&str)
	isString(*&str)
}
