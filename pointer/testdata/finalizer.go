package main

import "runtime"

func final1a(x *int) int {
	print(x) // @pointsto alloc@newint:10
	return *x
}

func final1b(x *bool) {
	print(x) // @pointsto
}

func setfinalizer1() {
	x := new(int)                    // @line newint
	runtime.SetFinalizer(x, final1a) // ok: final1a's result is ignored
	runtime.SetFinalizer(x, final1b) // param type mismatch: no effect
}

// @calls runtime.SetFinalizer -> main.final1a
// @calls main.setfinalizer1 -> runtime.SetFinalizer

func final2a(x *bool) {
	print(x) // @pointsto alloc@newbool1:10 | alloc@newbool2:10
}

func final2b(x *bool) {
	print(x) // @pointsto alloc@newbool1:10 | alloc@newbool2:10
}

func setfinalizer2() {
	x := new(bool) // @line newbool1
	f := final2a
	if unknown {
		x = new(bool) // @line newbool2
		f = final2b
	}
	runtime.SetFinalizer(x, f)
}

// @calls runtime.SetFinalizer -> main.final2a
// @calls runtime.SetFinalizer -> main.final2b
// @calls main.setfinalizer2 -> runtime.SetFinalizer

// type T int

// func (t *T) finalize() {
// 	print(t) // #@pointsto x
// }

// func setfinalizer3() {
// 	x := new(T)
// 	runtime.SetFinalizer(x, (*T).finalize) // go/types gives wrong type to f.
// }

// #@calls runtime.SetFinalizer -> (*T) finalize

func funcForPC() {
	f := runtime.FuncForPC(0) // @line funcforpc
	print(f)                  // @pointsto reflectAlloc@funcforpc:25
}

func main() {
	setfinalizer1()
	setfinalizer2()
	// setfinalizer3()
	funcForPC()
}

var unknown bool // defeat dead-code elimination
