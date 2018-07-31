// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm

package js_test

import (
	"fmt"
	"math"
	"syscall/js"
	"testing"
)

var dummys = js.Global().Call("eval", `({
	someBool: true,
	someString: "abc\u1234",
	someInt: 42,
	someFloat: 42.123,
	someArray: [41, 42, 43],
	add: function(a, b) {
		return a + b;
	},
	NaN: NaN,
})`)

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
	if dummys.Get("someBool") != dummys.Get("someBool") {
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
	if dummys.Get("someString") != dummys.Get("someString") {
		t.Errorf("same value not equal")
	}

	wantInt := "42"
	o = dummys.Get("someInt")
	if got := o.String(); got != wantInt {
		t.Errorf("got %#v, want %#v", got, wantInt)
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
	if dummys.Get("someInt") != dummys.Get("someInt") {
		t.Errorf("same value not equal")
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
	if dummys.Get("someFloat") != dummys.Get("someFloat") {
		t.Errorf("same value not equal")
	}
}

func TestObject(t *testing.T) {
	if dummys.Get("someArray") != dummys.Get("someArray") {
		t.Errorf("same value not equal")
	}

	// An object and its prototype should not be equal.
	proto := js.Global().Get("Object").Get("prototype")
	o := js.Global().Call("eval", "new Object()")
	if proto == o {
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

func TestTypedArrayOf(t *testing.T) {
	testTypedArrayOf(t, "[]int8", []int8{0, -42, 0}, -42)
	testTypedArrayOf(t, "[]int16", []int16{0, -42, 0}, -42)
	testTypedArrayOf(t, "[]int32", []int32{0, -42, 0}, -42)
	testTypedArrayOf(t, "[]uint8", []uint8{0, 42, 0}, 42)
	testTypedArrayOf(t, "[]uint16", []uint16{0, 42, 0}, 42)
	testTypedArrayOf(t, "[]uint32", []uint32{0, 42, 0}, 42)
	testTypedArrayOf(t, "[]float32", []float32{0, -42.5, 0}, -42.5)
	testTypedArrayOf(t, "[]float64", []float64{0, -42.5, 0}, -42.5)
}

func testTypedArrayOf(t *testing.T, name string, slice interface{}, want float64) {
	t.Run(name, func(t *testing.T) {
		a := js.TypedArrayOf(slice)
		got := a.Index(1).Float()
		a.Release()
		if got != want {
			t.Errorf("got %#v, want %#v", got, want)
		}
	})
}

func TestNaN(t *testing.T) {
	want := js.ValueOf(math.NaN())
	got := dummys.Get("NaN")
	if got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}
}

func TestUndefined(t *testing.T) {
	dummys.Set("test", js.Undefined())
	if dummys == js.Undefined() || dummys.Get("test") != js.Undefined() || dummys.Get("xyz") != js.Undefined() {
		t.Errorf("js.Undefined expected")
	}
}

func TestNull(t *testing.T) {
	dummys.Set("test1", nil)
	dummys.Set("test2", js.Null())
	if dummys == js.Null() || dummys.Get("test1") != js.Null() || dummys.Get("test2") != js.Null() {
		t.Errorf("js.Null expected")
	}
}

func TestLength(t *testing.T) {
	if got := dummys.Get("someArray").Length(); got != 3 {
		t.Errorf("got %#v, want %#v", got, 3)
	}
}

func TestIndex(t *testing.T) {
	if got := dummys.Get("someArray").Index(1).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
}

func TestSetIndex(t *testing.T) {
	dummys.Get("someArray").SetIndex(2, 99)
	if got := dummys.Get("someArray").Index(2).Int(); got != 99 {
		t.Errorf("got %#v, want %#v", got, 99)
	}
}

func TestCall(t *testing.T) {
	var i int64 = 40
	if got := dummys.Call("add", i, 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
	if got := dummys.Call("add", js.Global().Call("eval", "40"), 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
}

func TestInvoke(t *testing.T) {
	var i int64 = 40
	if got := dummys.Get("add").Invoke(i, 2).Int(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
}

func TestNew(t *testing.T) {
	if got := js.Global().Get("Array").New(42).Length(); got != 42 {
		t.Errorf("got %#v, want %#v", got, 42)
	}
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

type object = map[string]interface{}
type array = []interface{}

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

func TestCallback(t *testing.T) {
	c := make(chan struct{})
	cb := js.NewCallback(func(args []js.Value) {
		if got := args[0].Int(); got != 42 {
			t.Errorf("got %#v, want %#v", got, 42)
		}
		c <- struct{}{}
	})
	defer cb.Release()
	js.Global().Call("setTimeout", cb, 0, 42)
	<-c
}

func TestEventCallback(t *testing.T) {
	for _, name := range []string{"preventDefault", "stopPropagation", "stopImmediatePropagation"} {
		c := make(chan struct{})
		var flags js.EventCallbackFlag
		switch name {
		case "preventDefault":
			flags = js.PreventDefault
		case "stopPropagation":
			flags = js.StopPropagation
		case "stopImmediatePropagation":
			flags = js.StopImmediatePropagation
		}
		cb := js.NewEventCallback(flags, func(event js.Value) {
			c <- struct{}{}
		})
		defer cb.Release()

		event := js.Global().Call("eval", fmt.Sprintf("({ called: false, %s: function() { this.called = true; } })", name))
		cb.Invoke(event)
		if !event.Get("called").Bool() {
			t.Errorf("%s not called", name)
		}

		<-c
	}
}

func ExampleNewCallback() {
	var cb js.Callback
	cb = js.NewCallback(func(args []js.Value) {
		fmt.Println("button clicked")
		cb.Release() // release the callback if the button will not be clicked again
	})
	js.Global().Get("document").Call("getElementById", "myButton").Call("addEventListener", "click", cb)
}
