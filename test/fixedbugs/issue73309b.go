// compile

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// a Validator instance
type Validator []Validable

type Numeric interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

func (v Validator) Valid() bool {
	for _, field := range v {
		if !field.Validate() {
			return false
		}
	}
	return true
}

type Validable interface {
	Validate() bool
}

type FieldDef[T any] struct {
	value T
	rules []Rule[T]
}

func (f FieldDef[T]) Validate() bool {
	for _, rule := range f.rules {
		if !rule(f) {
			return false
		}
	}
	return true
}

type Rule[T any] = func(FieldDef[T]) bool

func Field[T any](value T, rules ...Rule[T]) *FieldDef[T] {
	return &FieldDef[T]{value: value, rules: rules}
}

type StringRule = Rule[string]

type NumericRule[T Numeric] = Rule[T]

type UnsignedRule[T Unsigned] = Rule[T]

func MinS(n int) StringRule {
	return func(fd FieldDef[string]) bool {
		return len(fd.value) < n
	}
}

func MinD[T Numeric](n T) NumericRule[T] {
	return func(fd FieldDef[T]) bool {
		return fd.value < n
	}
}

func MinU[T Unsigned](n T) UnsignedRule[T] {
	return func(fd FieldDef[T]) bool {
		return fd.value < n
	}
}

func main() {
	v := Validator{
		Field("test", MinS(5)),
	}

	if !v.Valid() {
		println("invalid")
		return
	}

	println("valid")
}
