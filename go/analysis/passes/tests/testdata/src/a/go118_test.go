//go:build go1.18
// +build go1.18

package a

import (
	"testing"
)

func Fuzzfoo(*testing.F) {} // want "first letter after 'Fuzz' must not be lowercase"

func FuzzBoo(*testing.F) {} // OK because first letter after 'Fuzz' is Uppercase.

func FuzzCallDifferentFunc(f *testing.F) {
	f.Name() //OK
}

func FuzzFunc(f *testing.F) {
	f.Fuzz(func(t *testing.T) {}) // OK "first argument is of type *testing.T"
}

func FuzzFuncWithArgs(f *testing.F) {
	f.Add()                                      // want `wrong number of values in call to \(\*testing.F\)\.Add: 0, fuzz target expects 2`
	f.Add(1, 2, 3, 4)                            // want `wrong number of values in call to \(\*testing.F\)\.Add: 4, fuzz target expects 2`
	f.Add(5, 5)                                  // want `mismatched type in call to \(\*testing.F\)\.Add: int, fuzz target expects \[\]byte`
	f.Add([]byte("hello"), 5)                    // want `mismatched types in call to \(\*testing.F\)\.Add: \[\[\]byte int\], fuzz target expects \[int \[\]byte\]`
	f.Add(5, []byte("hello"))                    // OK
	f.Fuzz(func(t *testing.T, i int, b []byte) { // OK "arguments in func are allowed"
		f.Add(5, []byte("hello"))     // want `fuzz target must not call any \*F methods`
		f.Name()                      // OK "calls to (*F).Failed and (*F).Name are allowed"
		f.Failed()                    // OK "calls to (*F).Failed and (*F).Name are allowed"
		f.Fuzz(func(t *testing.T) {}) // want `fuzz target must not call any \*F methods`
	})
}

func FuzzArgFunc(f *testing.F) {
	f.Fuzz(0) // want "argument to Fuzz must be a function"
}

func FuzzFuncWithReturn(f *testing.F) {
	f.Fuzz(func(t *testing.T) bool { return true }) // want "fuzz target must not return any value"
}

func FuzzFuncNoArg(f *testing.F) {
	f.Fuzz(func() {}) // want "fuzz target must have 1 or more argument"
}

func FuzzFuncFirstArgNotTesting(f *testing.F) {
	f.Fuzz(func(i int64) {}) // want "the first parameter of a fuzz target must be \\*testing.T"
}

func FuzzFuncFirstArgTestingNotT(f *testing.F) {
	f.Fuzz(func(t *testing.F) {}) // want "the first parameter of a fuzz target must be \\*testing.T"
}

func FuzzFuncSecondArgNotAllowed(f *testing.F) {
	f.Fuzz(func(t *testing.T, i complex64) {}) // want "fuzzing arguments can only have the following types: string, bool, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, \\[\\]byte"
}

func FuzzFuncSecondArgArrNotAllowed(f *testing.F) {
	f.Fuzz(func(t *testing.T, i []int) {}) // want "fuzzing arguments can only have the following types: string, bool, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, \\[\\]byte"
}

func FuzzFuncConsecutiveArgNotAllowed(f *testing.F) {
	f.Fuzz(func(t *testing.T, i, j string, k complex64) {}) // want "fuzzing arguments can only have the following types: string, bool, float32, float64, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, \\[\\]byte"
}

func FuzzFuncInner(f *testing.F) {
	innerFunc := func(t *testing.T, i float32) {}
	f.Fuzz(innerFunc) // ok
}

func FuzzArrayOfFunc(f *testing.F) {
	var funcs = []func(t *testing.T, i int){func(t *testing.T, i int) {}}
	f.Fuzz(funcs[0]) // ok
}

type GenericSlice[T any] []T

func FuzzGenericFunc(f *testing.F) {
	g := GenericSlice[func(t *testing.T, i int)]{func(t *testing.T, i int) {}}
	f.Fuzz(g[0]) // ok
}

type F func(t *testing.T, i int32)

type myType struct {
	myVar F
}

func FuzzObjectMethod(f *testing.F) {
	obj := myType{
		myVar: func(t *testing.T, i int32) {},
	}
	f.Fuzz(obj.myVar) // ok
}
