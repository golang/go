//go:build ignore
// +build ignore

package main

// This file is the input to TestValueForExpr in source_test.go, which
// ensures that each expression e immediately following a /*@kind*/(x)
// annotation, when passed to Function.ValueForExpr(e), returns a
// non-nil Value of the same type as e and of kind 'kind'.

func f(spilled, unspilled int) {
	_ = /*@UnOp*/ (spilled)
	_ = /*@Parameter*/ (unspilled)
	_ = /*@nil*/ (1 + 2) // (constant)
	i := 0

	f := func() (int, int) { return 0, 0 }

	/*@Call*/
	(print( /*@BinOp*/ (i + 1)))
	_, _ = /*@Call*/ (f())
	ch := /*@MakeChan*/ (make(chan int))
	/*@UnOp*/ (<-ch)
	x := /*@UnOp*/ (<-ch)
	_ = x
	select {
	case /*@Extract*/ (<-ch):
	case x := /*@Extract*/ (<-ch):
		_ = x
	}
	defer /*@Function*/ (func() {
	})()
	go /*@Function*/ (func() {
	})()
	y := 0
	if true && /*@BinOp*/ (bool(y > 0)) {
		y = 1
	}
	_ = /*@Phi*/ (y)
	map1 := /*@MakeMap*/ (make(map[string]string))
	_ = map1
	_ = /*@Slice*/ (make([]int, 0))
	_ = /*@MakeClosure*/ (func() { print(spilled) })

	sl := []int{}
	_ = /*@Slice*/ (sl[:0])

	_ = /*@nil*/ (new(int)) // optimized away
	tmp := /*@Alloc*/ (new(int))
	_ = tmp
	var iface interface{}
	_ = /*@TypeAssert*/ (iface.(int))
	_ = /*@UnOp*/ (sl[0])
	_ = /*@IndexAddr*/ (&sl[0])
	_ = /*@Index*/ ([2]int{}[0])
	var p *int
	_ = /*@UnOp*/ (*p)

	_ = /*@UnOp*/ (global)
	/*@UnOp*/ (global)[""] = ""
	/*@Global*/ (global) = map[string]string{}

	var local t
	/*UnOp*/ (local.x) = 1

	// Exercise corner-cases of lvalues vs rvalues.
	type N *N
	var n N
	/*@UnOp*/ (n) = /*@UnOp*/ (n)
	/*@ChangeType*/ (n) = /*@Alloc*/ (&n)
	/*@UnOp*/ (n) = /*@UnOp*/ (*n)
	/*@UnOp*/ (n) = /*@UnOp*/ (**n)
}

func complit() {
	// Composite literals.
	// We get different results for
	// - composite literal as value (e.g. operand to print)
	// - composite literal initializer for addressable value
	// - composite literal value assigned to blank var

	// 1. Slices
	print( /*@Slice*/ ([]int{}))
	print( /*@Alloc*/ (&[]int{}))
	print(& /*@Slice*/ ([]int{}))

	sl1 := /*@Slice*/ ([]int{})
	sl2 := /*@Alloc*/ (&[]int{})
	sl3 := & /*@Slice*/ ([]int{})
	_, _, _ = sl1, sl2, sl3

	_ = /*@Slice*/ ([]int{})
	_ = /*@nil*/ (& /*@Slice*/ ([]int{})) // & optimized away
	_ = & /*@Slice*/ ([]int{})

	// 2. Arrays
	print( /*@Const*/ ([1]int{}))
	print( /*@Alloc*/ (&[1]int{}))
	print(& /*@Alloc*/ ([1]int{}))

	arr1 := /*@Const*/ ([1]int{})
	arr2 := /*@Alloc*/ (&[1]int{})
	arr3 := & /*@Alloc*/ ([1]int{})
	_, _, _ = arr1, arr2, arr3

	_ = /*@Const*/ ([1]int{})
	_ = /*@nil*/ (& /*@Const*/ ([1]int{})) // & optimized away
	_ = & /*@Const*/ ([1]int{})

	// 3. Maps
	type M map[int]int
	print( /*@MakeMap*/ (M{}))
	print( /*@Alloc*/ (&M{}))
	print(& /*@MakeMap*/ (M{}))

	m1 := /*@MakeMap*/ (M{})
	m2 := /*@Alloc*/ (&M{})
	m3 := & /*@MakeMap*/ (M{})
	_, _, _ = m1, m2, m3

	_ = /*@MakeMap*/ (M{})
	_ = /*@nil*/ (& /*@MakeMap*/ (M{})) // & optimized away
	_ = & /*@MakeMap*/ (M{})

	// 4. Structs
	print( /*@Const*/ (struct{}{}))
	print( /*@Alloc*/ (&struct{}{}))
	print(& /*@Alloc*/ (struct{}{}))

	s1 := /*@Const*/ (struct{}{})
	s2 := /*@Alloc*/ (&struct{}{})
	s3 := & /*@Alloc*/ (struct{}{})
	_, _, _ = s1, s2, s3

	_ = /*@Const*/ (struct{}{})
	_ = /*@nil*/ (& /*@Const*/ (struct{}{})) // & optimized away
	_ = & /*@Const*/ (struct{}{})
}

type t struct{ x int }

// Ensure we can locate methods of named types.
func (t) f(param int) {
	_ = /*@Parameter*/ (param)
}

// Ensure we can locate init functions.
func init() {
	m := /*@MakeMap*/ (make(map[string]string))
	_ = m
}

// Ensure we can locate variables in initializer expressions.
var global = /*@MakeMap*/ (make(map[string]string))
