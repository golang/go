// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Test case from issue.

type Nat /* ERROR "invalid recursive type" */ interface {
	Zero|Succ
}

type Zero struct{}
type Succ struct{
	Nat // Nat contains type constraints but is invalid, so no error
}

// Struct tests.

type I1 interface {
	comparable
}

type I2 interface {
	~int
}

type I3 interface {
	I1
	I2
}

type _ struct {
	f I1 // ERRORx `interface is .* comparable`
}

type _ struct {
	comparable // ERRORx `interface is .* comparable`
}

type _ struct{
	I1 // ERRORx `interface is .* comparable`
}

type _ struct{
	I2 // ERROR "interface contains type constraints"
}

type _ struct{
	I3 // ERROR "interface contains type constraints"
}

// General composite types.

type (
	_ [10]I1 // ERRORx `interface is .* comparable`
	_ [10]I2 // ERROR "interface contains type constraints"

	_ []I1 // ERRORx `interface is .* comparable`
	_ []I2 // ERROR "interface contains type constraints"

	_ *I3 // ERROR "interface contains type constraints"
	_ map[I1 /* ERRORx `interface is .* comparable` */ ]I2 // ERROR "interface contains type constraints"
	_ chan I3 // ERROR "interface contains type constraints"
	_ func(I1 /* ERRORx `interface is .* comparable` */ )
	_ func() I2 // ERROR "interface contains type constraints"
)

// Other cases.

var _ = [...]I3 /* ERROR "interface contains type constraints" */ {}

func _(x interface{}) {
	_ = x.(I3 /* ERROR "interface contains type constraints" */ )
}

type T1[_ any] struct{}
type T3[_, _, _ any] struct{}
var _ T1[I2 /* ERROR "interface contains type constraints" */ ]
var _ T3[int, I2 /* ERROR "interface contains type constraints" */ , float32]

func f1[_ any]() int { panic(0) }
var _ = f1[I2 /* ERROR "interface contains type constraints" */ ]()
func f3[_, _, _ any]() int { panic(0) }
var _ = f3[int, I2 /* ERROR "interface contains type constraints" */ , float32]()

func _(x interface{}) {
	switch x.(type) {
	case I2 /* ERROR "interface contains type constraints" */ :
	}
}
