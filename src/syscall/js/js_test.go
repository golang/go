// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js && wasm

// To run these tests:
//
// - Install Node
// - Add /path/to/go/misc/wasm to your $PATH (so that "go test" can find
//   "go_js_wasm_exec").
// - GOOS=js GOARCH=wasm go test
//
// See -exec in "go help test", and "go help run" for details.

package js_test

import (
	"fmt"
	"math"
	"runtime"
	"syscall/js"
	"testing"
)

var dummys = js.Global().Call("eval", `({
	someBool: true,
	someString: "abc\u1234",
	someInt: 42,
	someFloat: 42.123,
	someArray: [41, 42, 43],
	someDate: new Date(),
	add: function(a, b) {
		return a + b;
	},
	zero: 0,
	stringZero: "0",
	NaN: NaN,
	emptyObj: {},
	emptyArray: [],
	Infinity: Infinity,
	NegInfinity: -Infinity,
	objNumber0: new Number(0),
	objBooleanFalse: new Boolean(false),
})`)

//go:wasmimport _gotest add
func testAdd(uint32, uint32) uint32

func TestWasmImport(t *testing.T) {
	a := uint32(3)
	b := uint32(5)
	want := a + b
	if got := testAdd(a, b); got != want {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBool(t *testing.T) {
	want := true
	o := dummys.Get("someBool")
	if got := o.Bool(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	dummys.Set("otherBool", want)
	if got := dummys.Get("otherBool").Bool(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if !dummys.Get("someBool").Equal(dummys.Get("someBool")) {
		t.Errorf("same value not equal")
	}
}

func TestString(t *testing.T) {
	want := "abc\u1234"
	o := dummys.Get("someString")
	if got := o.String(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	dummys.Set("otherString", want)
	if got := dummys.Get("otherString").String(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if !dummys.Get("someString").Equal(dummys.Get("someString")) {
		t.Errorf("same value not equal")
	}

	if got, want := js.Undefined().String(), "<undefined>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.Null().String(), "<null>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.ValueOf(true).String(), "<boolean: true>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.ValueOf(42.5).String(), "<number: 42.5>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.Global().Call("Symbol").String(), "<symbol>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.Global().String(), "<object>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := js.Global().Get("setTimeout").String(), "<function>"; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestInt(t *testing.T) {
	want := 42
	o := dummys.Get("someInt")
	if got := o.Int(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	dummys.Set("otherInt", want)
	if got := dummys.Get("otherInt").Int(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if !dummys.Get("someInt").Equal(dummys.Get("someInt")) {
		t.Errorf("same value not equal")
	}
	if got := dummys.Get("zero").Int(); got != 0 {
		t.Errorf("got %#v, want %#v", got, 0)
	}
}

func TestIntConversion(t *testing.T) {
	testIntConversion(t, 0)
	testIntConversion(t, 1)
	testIntConversion(t, -1)
	testIntConversion(t, 1<<20)
	testIntConversion(t, -1<<20)
	testIntConversion(t, 1<<40)
	testIntConversion(t, -1<<40)
	testIntConversion(t, 1<<60)
	testIntConversion(t, -1<<60)
}

func testIntConversion(t *testing.T, want int) {
	if got := js.ValueOf(want).Int(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestFloat(t *testing.T) {
	want := 42.123
	o := dummys.Get("someFloat")
	if got := o.Float(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	dummys.Set("otherFloat", want)
	if got := dummys.Get("otherFloat").Float(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if !dummys.Get("someFloat").Equal(dummys.Get("someFloat")) {
		t.Errorf("same value not equal")
	}
}

func TestObject(t *testing.T) {
	if !dummys.Get("someArray").Equal(dummys.Get("someArray")) {
		t.Errorf("same value not equal")
	}

	// An object and its prototype should not be equal.
	proto := js.Global().Get("Object").Get("prototype")
	o := js.Global().Call("eval", "new Object()")
	if proto.Equal(o) {
		t.Errorf("object equals to its prototype")
	}
}

func TestFrozenObject(t *testing.T) {
	o := js.Global().Call("eval", "(function () { let o = new Object(); o.field = 5; Object.freeze(o); return o; })()")
	want := 5
	if got := o.Get("field").Int(); want != got {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestEqual(t *testing.T) {
	if !dummys.Get("someFloat").Equal(dummys.Get("someFloat")) {
		t.Errorf("same float is not equal")
	}
	if !dummys.Get("emptyObj").Equal(dummys.Get("emptyObj")) {
		t.Errorf("same object is not equal")
	}
	if dummys.Get("someFloat").Equal(dummys.Get("someInt")) {
		t.Errorf("different values are not unequal")
	}
}

func TestNaN(t *testing.T) {
	if !dummys.Get("NaN").IsNaN() {
		t.Errorf("JS NaN is not NaN")
	}
	if !js.ValueOf(math.NaN()).IsNaN() {
		t.Errorf("Go NaN is not NaN")
	}
	if dummys.Get("NaN").Equal(dummys.Get("NaN")) {
		t.Errorf("NaN is equal to NaN")
	}
}

func TestUndefined(t *testing.T) {
	if !js.Undefined().IsUndefined() {
		t.Errorf("undefined is not undefined")
	}
	if !js.Undefined().Equal(js.Undefined()) {
		t.Errorf("undefined is not equal to undefined")
	}
	if dummys.IsUndefined() {
		t.Errorf("object is undefined")
	}
	if js.Undefined().IsNull() {
		t.Errorf("undefined is null")
	}
	if dummys.Set("test", js.Undefined()); !dummys.Get("test").IsUndefined() {
		t.Errorf("could not set undefined")
	}
}

func TestNull(t *testing.T) {
	if !js.Null().IsNull() {
		t.Errorf("null is not null")
	}
	if !js.Null().Equal(js.Null()) {
		t.Errorf("null is not equal to null")
	}
	if dummys.IsNull() {
		t.Errorf("object is null")
	}
	if js.Null().IsUndefined() {
		t.Errorf("null is undefined")
	}
	if dummys.Set("test", js.Null()); !dummys.Get("test").IsNull() {
		t.Errorf("could not set null")
	}
	if dummys.Set("test", nil); !dummys.Get("test").IsNull() {
		t.Errorf("could not set nil")
	}
}

func TestLength(t *testing.T) {
	if got := dummys.Get("someArray").Length(); got != 3 {
		t.Errorf("got %#v, want %#v", got, 3)
	}
}

func TestGet(t *testing.T) {
	// positive cases get tested per type

	expectValueError(t, func() {
		dummys.Get("zero").Get("badField")
	})
}

func TestSet(t *testing.T) {
	// positive cases get tested per type

	expectValueError(t, func() {
		dummys.Get("zero").Set("badField", 42)
	})
}

func TestDelete(t *testing.T) {
	dummys.Set("test", 42)
	dummys.Delete("test")
	if dummys.Call("hasOwnProperty", "test").Bool() {
		t.Errorf("property still exists")
	}

	expectValueError(t, func() {
		dummys.Get("zero").Delete("badField")
	})
}

func TestIndex(t *testing.T) {
	if got := dummys.Get("someArray").Index(1).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}

	expectValueError(t, func() {
		dummys.Get("zero").Index(1)
	})
}

func TestSetIndex(t *testing.T) {
	dummys.Get("someArray").SetIndex(2, 99)
	if got := dummys.Get("someArray").Index(2).Int(); got != 99 {
		t.Errorf("got %#v, want %#v", got, 99)
	}

	expectValueError(t, func() {
		dummys.Get("zero").SetIndex(2, 99)
	})
}

func TestCall(t *testing.T) {
	var i int64 = 40
	if got := dummys.Call("add", i, 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
	if got := dummys.Call("add", js.Global().Call("eval", "40"), 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}

	expectPanic(t, func() {
		dummys.Call("zero")
	})
	expectValueError(t, func() {
		dummys.Get("zero").Call("badMethod")
	})
}

func TestInvoke(t *testing.T) {
	var i int64 = 40
	if got := dummys.Get("add").Invoke(i, 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}

	expectValueError(t, func() {
		dummys.Get("zero").Invoke()
	})
}

func TestNew(t *testing.T) {
	if got := js.Global().Get("Array").New(42).Length(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}

	expectValueError(t, func() {
		dummys.Get("zero").New()
	})
}

func TestInstanceOf(t *testing.T) {
	someArray := js.Global().Get("Array").New()
	if got, want := someArray.InstanceOf(js.Global().Get("Array")), true; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got, want := someArray.InstanceOf(js.Global().Get("Function")), false; got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestType(t *testing.T) {
	if got, want := js.Undefined().Type(), js.TypeUndefined; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.Null().Type(), js.TypeNull; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.ValueOf(true).Type(), js.TypeBoolean; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.ValueOf(0).Type(), js.TypeNumber; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.ValueOf(42).Type(), js.TypeNumber; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.ValueOf("test").Type(), js.TypeString; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.Global().Get("Symbol").Invoke("test").Type(), js.TypeSymbol; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.Global().Get("Array").New().Type(), js.TypeObject; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
	if got, want := js.Global().Get("Array").Type(), js.TypeFunction; got != want {
		t.Errorf("got %s, want %s", got, want)
	}
}

type object = map[string]any
type array = []any

func TestValueOf(t *testing.T) {
	a := js.ValueOf(array{0, array{0, 42, 0}, 0})
	if got := a.Index(1).Index(1).Int(); got != 42 {
		t.Errorf("got %v, want %v", got, 42)
	}

	o := js.ValueOf(object{"x": object{"y": 42}})
	if got := o.Get("x").Get("y").Int(); got != 42 {
		t.Errorf("got %v, want %v", got, 42)
	}
}

func TestZeroValue(t *testing.T) {
	var v js.Value
	if !v.IsUndefined() {
		t.Error("zero js.Value is not js.Undefined()")
	}
}

func TestFuncOf(t *testing.T) {
	c := make(chan struct{})
	cb := js.FuncOf(func(this js.Value, args []js.Value) any {
		if got := args[0].Int(); got != 42 {
			t.Errorf("got %#v, want %#v", got, 42)
		}
		c <- struct{}{}
		return nil
	})
	defer cb.Release()
	js.Global().Call("setTimeout", cb, 0, 42)
	<-c
}

func TestInvokeFunction(t *testing.T) {
	called := false
	cb := js.FuncOf(func(this js.Value, args []js.Value) any {
		cb2 := js.FuncOf(func(this js.Value, args []js.Value) any {
			called = true
			return 42
		})
		defer cb2.Release()
		return cb2.Invoke()
	})
	defer cb.Release()
	if got := cb.Invoke().Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
	if !called {
		t.Error("function not called")
	}
}

func TestInterleavedFunctions(t *testing.T) {
	c1 := make(chan struct{})
	c2 := make(chan struct{})

	js.Global().Get("setTimeout").Invoke(js.FuncOf(func(this js.Value, args []js.Value) any {
		c1 <- struct{}{}
		<-c2
		return nil
	}), 0)

	<-c1
	c2 <- struct{}{}
	// this goroutine is running, but the callback of setTimeout did not return yet, invoke another function now
	f := js.FuncOf(func(this js.Value, args []js.Value) any {
		return nil
	})
	f.Invoke()
}

func ExampleFuncOf() {
	var cb js.Func
	cb = js.FuncOf(func(this js.Value, args []js.Value) any {
		fmt.Println("button clicked")
		cb.Release() // release the function if the button will not be clicked again
		return nil
	})
	js.Global().Get("document").Call("getElementById", "myButton").Call("addEventListener", "click", cb)
}

// See
// - https://developer.mozilla.org/en-US/docs/Glossary/Truthy
// - https://stackoverflow.com/questions/19839952/all-falsey-values-in-javascript/19839953#19839953
// - http://www.ecma-international.org/ecma-262/5.1/#sec-9.2
func TestTruthy(t *testing.T) {
	want := true
	for _, key := range []string{
		"someBool", "someString", "someInt", "someFloat", "someArray", "someDate",
		"stringZero", // "0" is truthy
		"add",        // functions are truthy
		"emptyObj", "emptyArray", "Infinity", "NegInfinity",
		// All objects are truthy, even if they're Number(0) or Boolean(false).
		"objNumber0", "objBooleanFalse",
	} {
		if got := dummys.Get(key).Truthy(); got != want {
			t.Errorf("%s: got %#v, want %#v", key, got, want)
		}
	}

	want = false
	if got := dummys.Get("zero").Truthy(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got := dummys.Get("NaN").Truthy(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got := js.ValueOf("").Truthy(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got := js.Null().Truthy(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
	if got := js.Undefined().Truthy(); got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func expectValueError(t *testing.T, fn func()) {
	defer func() {
		err := recover()
		if _, ok := err.(*js.ValueError); !ok {
			t.Errorf("expected *js.ValueError, got %T", err)
		}
	}()
	fn()
}

func expectPanic(t *testing.T, fn func()) {
	defer func() {
		err := recover()
		if err == nil {
			t.Errorf("expected panic")
		}
	}()
	fn()
}

var copyTests = []struct {
	srcLen  int
	dstLen  int
	copyLen int
}{
	{5, 3, 3},
	{3, 5, 3},
	{0, 0, 0},
}

func TestCopyBytesToGo(t *testing.T) {
	for _, tt := range copyTests {
		t.Run(fmt.Sprintf("%d-to-%d", tt.srcLen, tt.dstLen), func(t *testing.T) {
			src := js.Global().Get("Uint8Array").New(tt.srcLen)
			if tt.srcLen >= 2 {
				src.SetIndex(1, 42)
			}
			dst := make([]byte, tt.dstLen)

			if got, want := js.CopyBytesToGo(dst, src), tt.copyLen; got != want {
				t.Errorf("copied %d, want %d", got, want)
			}
			if tt.dstLen >= 2 {
				if got, want := int(dst[1]), 42; got != want {
					t.Errorf("got %d, want %d", got, want)
				}
			}
		})
	}
}

func TestCopyBytesToJS(t *testing.T) {
	for _, tt := range copyTests {
		t.Run(fmt.Sprintf("%d-to-%d", tt.srcLen, tt.dstLen), func(t *testing.T) {
			src := make([]byte, tt.srcLen)
			if tt.srcLen >= 2 {
				src[1] = 42
			}
			dst := js.Global().Get("Uint8Array").New(tt.dstLen)

			if got, want := js.CopyBytesToJS(dst, src), tt.copyLen; got != want {
				t.Errorf("copied %d, want %d", got, want)
			}
			if tt.dstLen >= 2 {
				if got, want := dst.Index(1).Int(), 42; got != want {
					t.Errorf("got %d, want %d", got, want)
				}
			}
		})
	}
}

func TestGarbageCollection(t *testing.T) {
	before := js.JSGo.Get("_values").Length()
	for i := 0; i < 1000; i++ {
		_ = js.Global().Get("Object").New().Call("toString").String()
		runtime.GC()
	}
	after := js.JSGo.Get("_values").Length()
	if after-before > 500 {
		t.Errorf("garbage collection ineffective")
	}
}

// This table is used for allocation tests. We expect a specific allocation
// behavior to be seen, depending on the number of arguments applied to various
// JavaScript functions.
// Note: All JavaScript functions return a JavaScript array, which will cause
// one allocation to be created to track the Value.gcPtr for the Value finalizer.
var allocTests = []struct {
	argLen  int // The number of arguments to use for the syscall
	expected int // The expected number of allocations
}{
	// For less than or equal to 16 arguments, we expect 1 alloction:
	// - makeValue new(ref)
	{0,  1},
	{2,  1},
	{15, 1},
	{16, 1},
	// For greater than 16 arguments, we expect 3 alloction:
	// - makeValue: new(ref)
	// - makeArgSlices: argVals = make([]Value, size)
	// - makeArgSlices: argRefs = make([]ref, size)
	{17, 3},
	{32, 3},
	{42, 3},
}

// TestCallAllocations ensures the correct allocation profile for Value.Call
func TestCallAllocations(t *testing.T) {
	for _, test := range allocTests {
		args := make([]any, test.argLen)

		tmpArray := js.Global().Get("Array").New(0)
		numAllocs := testing.AllocsPerRun(100, func() {
			tmpArray.Call("concat", args...)
		});

		if numAllocs != float64(test.expected) {
			t.Errorf("got numAllocs %#v, want %#v", numAllocs, test.expected)
		}
	}
}

// TestInvokeAllocations ensures the correct allocation profile for Value.Invoke
func TestInvokeAllocations(t *testing.T) {
	for _, test := range allocTests {
		args := make([]any, test.argLen)

		tmpArray := js.Global().Get("Array").New(0)
		concatFunc := tmpArray.Get("concat").Call("bind", tmpArray)
		numAllocs := testing.AllocsPerRun(100, func() {
			concatFunc.Invoke(args...)
		});

		if numAllocs != float64(test.expected) {
			t.Errorf("got numAllocs %#v, want %#v", numAllocs, test.expected)
		}
	}
}

// TestNewAllocations ensures the correct allocation profile for Value.New
func TestNewAllocations(t *testing.T) {
	arrayConstructor := js.Global().Get("Array")

	for _, test := range allocTests {
		args := make([]any, test.argLen)

		numAllocs := testing.AllocsPerRun(100, func() {
			arrayConstructor.New(args...)
		});

		if numAllocs != float64(test.expected) {
			t.Errorf("got numAllocs %#v, want %#v", numAllocs, test.expected)
		}
	}
}

// BenchmarkDOM is a simple benchmark which emulates a webapp making DOM operations.
// It creates a div, and sets its id. Then searches by that id and sets some data.
// Finally it removes that div.
func BenchmarkDOM(b *testing.B) {
	document := js.Global().Get("document")
	if document.IsUndefined() {
		b.Skip("Not a browser environment. Skipping.")
	}
	const data = "someString"
	for i := 0; i < b.N; i++ {
		div := document.Call("createElement", "div")
		div.Call("setAttribute", "id", "myDiv")
		document.Get("body").Call("appendChild", div)
		myDiv := document.Call("getElementById", "myDiv")
		myDiv.Set("innerHTML", data)

		if got, want := myDiv.Get("innerHTML").String(), data; got != want {
			b.Errorf("got %s, want %s", got, want)
		}
		document.Get("body").Call("removeChild", div)
	}
}

func TestGlobal(t *testing.T) {
	ident := js.FuncOf(func(this js.Value, args []js.Value) any {
		return args[0]
	})
	defer ident.Release()

	if got := ident.Invoke(js.Global()); !got.Equal(js.Global()) {
		t.Errorf("got %#v, want %#v", got, js.Global())
	}
}
