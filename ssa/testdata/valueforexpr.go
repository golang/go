//+build ignore

package main

// This file is the input to TestValueForExpr in source_test.go, which
// ensures that each expression e immediately following a /*@kind*/(x)
// annotation, when passed to Function.ValueForExpr(e), returns a
// non-nil Value of the same type as e and of kind 'kind'.

func f(param int) {
	_ = /*@<nil>*/ (1 + 2) // (constant)
	i := 0
	/*@Call*/ (print( /*@BinOp*/ (i + 1)))
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
	_ = /*@MakeMap*/ (map[string]string{"": ""})
	_ = /*@MakeSlice*/ (make([]int, 0))
	_ = /*@MakeClosure*/ (func() { print(param) })
	sl := /*@Slice*/ ([]int{})
	_ = /*@Alloc*/ (&struct{}{})
	_ = /*@Slice*/ (sl[:0])
	_ = /*@Alloc*/ (new(int))
	var iface interface{}
	_ = /*@TypeAssert*/ (iface.(int))
	_ = /*@UnOp*/ (sl[0])
	_ = /*@IndexAddr*/ (&sl[0])
	_ = /*@Index*/ ([2]int{}[0])
	var p *int
	_ = /*@UnOp*/ (*p)
}
