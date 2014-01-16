package main

import "runtime"

func final1a(x *int) int {
	print(x) // @pointsto new@newint:10
	return *x
}

func final1b(x *bool) {
	print(x) // @pointsto
}

func runtimeSetFinalizer1() {
	x := new(int)                    // @line newint
	runtime.SetFinalizer(x, final1a) // ok: final1a's result is ignored
	runtime.SetFinalizer(x, final1b) // param type mismatch: no effect
}

// @calls main.runtimeSetFinalizer1 -> main.final1a
// @calls main.runtimeSetFinalizer1 -> main.final1b

func final2a(x *bool) {
	print(x) // @pointsto new@newbool1:10 | new@newbool2:10
}

func final2b(x *bool) {
	print(x) // @pointsto new@newbool1:10 | new@newbool2:10
}

func runtimeSetFinalizer2() {
	x := new(bool) // @line newbool1
	f := final2a
	if unknown {
		x = new(bool) // @line newbool2
		f = final2b
	}
	runtime.SetFinalizer(x, f)
}

// @calls main.runtimeSetFinalizer2 -> main.final2a
// @calls main.runtimeSetFinalizer2 -> main.final2b

type T int

func (t *T) finalize() {
	print(t) // @pointsto new@final3:10
}

func runtimeSetFinalizer3() {
	x := new(T) // @line final3
	runtime.SetFinalizer(x, (*T).finalize)
}

// @calls main.runtimeSetFinalizer3 -> (*main.T).finalize

// I hope I never live to see this code in the wild.
var setFinalizer = runtime.SetFinalizer

func final4(x *int) {
	print(x) // @pointsto new@finalIndirect:10
}

func runtimeSetFinalizerIndirect() {
	// In an indirect call, the shared contour for SetFinalizer is
	// used, i.e. the call is not inlined and appears in the call graph.
	x := new(int) // @line finalIndirect
	setFinalizer(x, final4)
}

// @calls main.runtimeSetFinalizerIndirect -> runtime.SetFinalizer
// @calls runtime.SetFinalizer -> main.final4

func main() {
	runtimeSetFinalizer1()
	runtimeSetFinalizer2()
	runtimeSetFinalizer3()
	runtimeSetFinalizerIndirect()
}

var unknown bool // defeat dead-code elimination
