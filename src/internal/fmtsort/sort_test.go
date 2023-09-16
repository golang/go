// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmtsort_test

import (
	"fmt"
	"internal/fmtsort"
	"math"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"
	"unsafe"
)

var compareTests = [][]reflect.Value{
	ct(reflect.TypeOf(int(0)), -1, 0, 1),
	ct(reflect.TypeOf(int8(0)), -1, 0, 1),
	ct(reflect.TypeOf(int16(0)), -1, 0, 1),
	ct(reflect.TypeOf(int32(0)), -1, 0, 1),
	ct(reflect.TypeOf(int64(0)), -1, 0, 1),
	ct(reflect.TypeOf(uint(0)), 0, 1, 5),
	ct(reflect.TypeOf(uint8(0)), 0, 1, 5),
	ct(reflect.TypeOf(uint16(0)), 0, 1, 5),
	ct(reflect.TypeOf(uint32(0)), 0, 1, 5),
	ct(reflect.TypeOf(uint64(0)), 0, 1, 5),
	ct(reflect.TypeOf(uintptr(0)), 0, 1, 5),
	ct(reflect.TypeOf(string("")), "", "a", "ab"),
	ct(reflect.TypeOf(float32(0)), math.NaN(), math.Inf(-1), -1e10, 0, 1e10, math.Inf(1)),
	ct(reflect.TypeOf(float64(0)), math.NaN(), math.Inf(-1), -1e10, 0, 1e10, math.Inf(1)),
	ct(reflect.TypeOf(complex64(0+1i)), -1-1i, -1+0i, -1+1i, 0-1i, 0+0i, 0+1i, 1-1i, 1+0i, 1+1i),
	ct(reflect.TypeOf(complex128(0+1i)), -1-1i, -1+0i, -1+1i, 0-1i, 0+0i, 0+1i, 1-1i, 1+0i, 1+1i),
	ct(reflect.TypeOf(false), false, true),
	ct(reflect.TypeOf(&ints[0]), &ints[0], &ints[1], &ints[2]),
	ct(reflect.TypeOf(unsafe.Pointer(&ints[0])), unsafe.Pointer(&ints[0]), unsafe.Pointer(&ints[1]), unsafe.Pointer(&ints[2])),
	ct(reflect.TypeOf(chans[0]), chans[0], chans[1], chans[2]),
	ct(reflect.TypeOf(toy{}), toy{0, 1}, toy{0, 2}, toy{1, -1}, toy{1, 1}),
	ct(reflect.TypeOf([2]int{}), [2]int{1, 1}, [2]int{1, 2}, [2]int{2, 0}),
	ct(reflect.TypeOf(any(0)), iFace, 1, 2, 3),
}

var iFace any

func ct(typ reflect.Type, args ...any) []reflect.Value {
	value := make([]reflect.Value, len(args))
	for i, v := range args {
		x := reflect.ValueOf(v)
		if !x.IsValid() { // Make it a typed nil.
			x = reflect.Zero(typ)
		} else {
			x = x.Convert(typ)
		}
		value[i] = x
	}
	return value
}

func TestCompare(t *testing.T) {
	for _, test := range compareTests {
		for i, v0 := range test {
			for j, v1 := range test {
				c := fmtsort.Compare(v0, v1)
				var expect int
				switch {
				case i == j:
					expect = 0
					// NaNs are tricky.
					if typ := v0.Type(); (typ.Kind() == reflect.Float32 || typ.Kind() == reflect.Float64) && math.IsNaN(v0.Float()) {
						expect = -1
					}
				case i < j:
					expect = -1
				case i > j:
					expect = 1
				}
				if c != expect {
					t.Errorf("%s: compare(%v,%v)=%d; expect %d", v0.Type(), v0, v1, c, expect)
				}
			}
		}
	}
}

type sortTest struct {
	data  any    // Always a map.
	print string // Printed result using our custom printer.
}

var sortTests = []sortTest{
	{
		map[int]string{7: "bar", -3: "foo"},
		"-3:foo 7:bar",
	},
	{
		map[uint8]string{7: "bar", 3: "foo"},
		"3:foo 7:bar",
	},
	{
		map[string]string{"7": "bar", "3": "foo"},
		"3:foo 7:bar",
	},
	{
		map[float64]string{7: "bar", -3: "foo", math.NaN(): "nan", math.Inf(0): "inf"},
		"NaN:nan -3:foo 7:bar +Inf:inf",
	},
	{
		map[complex128]string{7 + 2i: "bar2", 7 + 1i: "bar", -3: "foo", complex(math.NaN(), 0i): "nan", complex(math.Inf(0), 0i): "inf"},
		"(NaN+0i):nan (-3+0i):foo (7+1i):bar (7+2i):bar2 (+Inf+0i):inf",
	},
	{
		map[bool]string{true: "true", false: "false"},
		"false:false true:true",
	},
	{
		chanMap(),
		"CHAN0:0 CHAN1:1 CHAN2:2",
	},
	{
		pointerMap(),
		"PTR0:0 PTR1:1 PTR2:2",
	},
	{
		unsafePointerMap(),
		"UNSAFEPTR0:0 UNSAFEPTR1:1 UNSAFEPTR2:2",
	},
	{
		map[toy]string{{7, 2}: "72", {7, 1}: "71", {3, 4}: "34"},
		"{3 4}:34 {7 1}:71 {7 2}:72",
	},
	{
		map[[2]int]string{{7, 2}: "72", {7, 1}: "71", {3, 4}: "34"},
		"[3 4]:34 [7 1]:71 [7 2]:72",
	},
}

func sprint(data any) string {
	om := fmtsort.Sort(reflect.ValueOf(data))
	if om == nil {
		return "nil"
	}
	b := new(strings.Builder)
	for i, key := range om.Key {
		if i > 0 {
			b.WriteRune(' ')
		}
		b.WriteString(sprintKey(key))
		b.WriteRune(':')
		fmt.Fprint(b, om.Value[i])
	}
	return b.String()
}

// sprintKey formats a reflect.Value but gives reproducible values for some
// problematic types such as pointers. Note that it only does special handling
// for the troublesome types used in the test cases; it is not a general
// printer.
func sprintKey(key reflect.Value) string {
	switch str := key.Type().String(); str {
	case "*int":
		ptr := key.Interface().(*int)
		for i := range ints {
			if ptr == &ints[i] {
				return fmt.Sprintf("PTR%d", i)
			}
		}
		return "PTR???"
	case "unsafe.Pointer":
		ptr := key.Interface().(unsafe.Pointer)
		for i := range ints {
			if ptr == unsafe.Pointer(&ints[i]) {
				return fmt.Sprintf("UNSAFEPTR%d", i)
			}
		}
		return "UNSAFEPTR???"
	case "chan int":
		c := key.Interface().(chan int)
		for i := range chans {
			if c == chans[i] {
				return fmt.Sprintf("CHAN%d", i)
			}
		}
		return "CHAN???"
	default:
		return fmt.Sprint(key)
	}
}

var (
	ints  [3]int
	chans = makeChans()
	pin   runtime.Pinner
)

func makeChans() []chan int {
	cs := []chan int{make(chan int), make(chan int), make(chan int)}
	// Order channels by address. See issue #49431.
	for i := range cs {
		pin.Pin(reflect.ValueOf(cs[i]).UnsafePointer())
	}
	sort.Slice(cs, func(i, j int) bool {
		return uintptr(reflect.ValueOf(cs[i]).UnsafePointer()) < uintptr(reflect.ValueOf(cs[j]).UnsafePointer())
	})
	return cs
}

func pointerMap() map[*int]string {
	m := make(map[*int]string)
	for i := 2; i >= 0; i-- {
		m[&ints[i]] = fmt.Sprint(i)
	}
	return m
}

func unsafePointerMap() map[unsafe.Pointer]string {
	m := make(map[unsafe.Pointer]string)
	for i := 2; i >= 0; i-- {
		m[unsafe.Pointer(&ints[i])] = fmt.Sprint(i)
	}
	return m
}

func chanMap() map[chan int]string {
	m := make(map[chan int]string)
	for i := 2; i >= 0; i-- {
		m[chans[i]] = fmt.Sprint(i)
	}
	return m
}

type toy struct {
	A int // Exported.
	b int // Unexported.
}

func TestOrder(t *testing.T) {
	for _, test := range sortTests {
		got := sprint(test.data)
		if got != test.print {
			t.Errorf("%s: got %q, want %q", reflect.TypeOf(test.data), got, test.print)
		}
	}
}

func TestInterface(t *testing.T) {
	// A map containing multiple concrete types should be sorted by type,
	// then value. However, the relative ordering of types is unspecified,
	// so test this by checking the presence of sorted subgroups.
	m := map[any]string{
		[2]int{1, 0}:             "",
		[2]int{0, 1}:             "",
		true:                     "",
		false:                    "",
		3.1:                      "",
		2.1:                      "",
		1.1:                      "",
		math.NaN():               "",
		3:                        "",
		2:                        "",
		1:                        "",
		"c":                      "",
		"b":                      "",
		"a":                      "",
		struct{ x, y int }{1, 0}: "",
		struct{ x, y int }{0, 1}: "",
	}
	got := sprint(m)
	typeGroups := []string{
		"NaN: 1.1: 2.1: 3.1:", // float64
		"false: true:",        // bool
		"1: 2: 3:",            // int
		"a: b: c:",            // string
		"[0 1]: [1 0]:",       // [2]int
		"{0 1}: {1 0}:",       // struct{ x int; y int }
	}
	for _, g := range typeGroups {
		if !strings.Contains(got, g) {
			t.Errorf("sorted map should contain %q", g)
		}
	}
}
