// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect_test

import (
	"fmt"
	. "reflect"
	"strconv"
	"testing"
)

var sourceAll = struct {
	Bool         Value
	String       Value
	Bytes        Value
	NamedBytes   Value
	BytesArray   Value
	SliceAny     Value
	MapStringAny Value
}{
	Bool:         ValueOf(new(bool)).Elem(),
	String:       ValueOf(new(string)).Elem(),
	Bytes:        ValueOf(new([]byte)).Elem(),
	NamedBytes:   ValueOf(new(namedBytes)).Elem(),
	BytesArray:   ValueOf(new([32]byte)).Elem(),
	SliceAny:     ValueOf(new([]any)).Elem(),
	MapStringAny: ValueOf(new(map[string]any)).Elem(),
}

var sinkAll struct {
	RawBool   bool
	RawString string
	RawBytes  []byte
	RawInt    int
}

func BenchmarkBool(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawBool = sourceAll.Bool.Bool()
	}
}

func BenchmarkString(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawString = sourceAll.String.String()
	}
}

func BenchmarkBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawBytes = sourceAll.Bytes.Bytes()
	}
}

func BenchmarkNamedBytes(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawBytes = sourceAll.NamedBytes.Bytes()
	}
}

func BenchmarkBytesArray(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawBytes = sourceAll.BytesArray.Bytes()
	}
}

func BenchmarkSliceLen(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawInt = sourceAll.SliceAny.Len()
	}
}

func BenchmarkMapLen(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawInt = sourceAll.MapStringAny.Len()
	}
}

func BenchmarkStringLen(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawInt = sourceAll.String.Len()
	}
}

func BenchmarkArrayLen(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawInt = sourceAll.BytesArray.Len()
	}
}

func BenchmarkSliceCap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		sinkAll.RawInt = sourceAll.SliceAny.Cap()
	}
}

func BenchmarkDeepEqual(b *testing.B) {
	for _, bb := range deepEqualPerfTests {
		b.Run(ValueOf(bb.x).Type().String(), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				sink = DeepEqual(bb.x, bb.y)
			}
		})
	}
}

func BenchmarkMapsDeepEqual(b *testing.B) {
	m1 := map[int]int{
		1: 1, 2: 2,
	}
	m2 := map[int]int{
		1: 1, 2: 2,
	}
	for i := 0; i < b.N; i++ {
		DeepEqual(m1, m2)
	}
}

func BenchmarkBigSliceDeepEqual(b *testing.B) {
	var s1, s2 = make([]int16, 10240), make([]int16, 10240)
	s2[2] = 9
	for i := 0; i < b.N; i++ {
		DeepEqual(s1, s2)
	}
}

func BenchmarkIsZero(b *testing.B) {
	type Int4 struct {
		a, b, c, d int
	}
	type Int1024 struct {
		a [1024]int
	}
	type Int512 struct {
		a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16 [16]S
	}
	s := struct {
		ArrayComparable      [4]T
		ArrayIncomparable    [4]_Complex
		StructComparable     T
		StructIncomparable   _Complex
		ArrayInt_4           [4]int
		ArrayInt_1024        [1024]int
		ArrayInt_1024_NoZero [1024]int
		Struct4Int           Int4
		ArrayStruct4Int_1024 [256]Int4
		ArrayChanInt_1024    [1024]chan int
		StructInt_512        Int512
	}{}
	s.ArrayInt_1024_NoZero[512] = 1
	source := ValueOf(s)

	for i := 0; i < source.NumField(); i++ {
		name := source.Type().Field(i).Name
		value := source.Field(i)
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				sink = value.IsZero()
			}
		})
	}
}

func BenchmarkSetZero(b *testing.B) {
	source := ValueOf(new(struct {
		Bool      bool
		Int       int64
		Uint      uint64
		Float     float64
		Complex   complex128
		Array     [4]Value
		Chan      chan Value
		Func      func() Value
		Interface interface{ String() string }
		Map       map[string]Value
		Pointer   *Value
		Slice     []Value
		String    string
		Struct    Value
	})).Elem()

	for i := 0; i < source.NumField(); i++ {
		name := source.Type().Field(i).Name
		value := source.Field(i)
		zero := Zero(value.Type())
		b.Run(name+"/Direct", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				value.SetZero()
			}
		})
		b.Run(name+"/CachedZero", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				value.Set(zero)
			}
		})
		b.Run(name+"/NewZero", func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				value.Set(Zero(value.Type()))
			}
		})
	}
}

func BenchmarkSelect(b *testing.B) {
	channel := make(chan int)
	close(channel)
	var cases []SelectCase
	for i := 0; i < 8; i++ {
		cases = append(cases, SelectCase{
			Dir:  SelectRecv,
			Chan: ValueOf(channel),
		})
	}
	for _, numCases := range []int{1, 4, 8} {
		b.Run(strconv.Itoa(numCases), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				_, _, _ = Select(cases[:numCases])
			}
		})
	}
}

func BenchmarkCall(b *testing.B) {
	fv := ValueOf(func(a, b string) {})
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		args := []Value{ValueOf("a"), ValueOf("b")}
		for pb.Next() {
			fv.Call(args)
		}
	})
}

type myint int64

func (i *myint) inc() {
	*i = *i + 1
}

func BenchmarkCallMethod(b *testing.B) {
	b.ReportAllocs()
	z := new(myint)

	v := ValueOf(z.inc)
	for i := 0; i < b.N; i++ {
		v.Call(nil)
	}
}

func BenchmarkCallArgCopy(b *testing.B) {
	byteArray := func(n int) Value {
		return Zero(ArrayOf(n, TypeOf(byte(0))))
	}
	sizes := [...]struct {
		fv  Value
		arg Value
	}{
		{ValueOf(func(a [128]byte) {}), byteArray(128)},
		{ValueOf(func(a [256]byte) {}), byteArray(256)},
		{ValueOf(func(a [1024]byte) {}), byteArray(1024)},
		{ValueOf(func(a [4096]byte) {}), byteArray(4096)},
		{ValueOf(func(a [65536]byte) {}), byteArray(65536)},
	}
	for _, size := range sizes {
		bench := func(b *testing.B) {
			args := []Value{size.arg}
			b.SetBytes(int64(size.arg.Len()))
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					size.fv.Call(args)
				}
			})
		}
		name := fmt.Sprintf("size=%v", size.arg.Len())
		b.Run(name, bench)
	}
}

func BenchmarkPtrTo(b *testing.B) {
	// Construct a type with a zero ptrToThis.
	type T struct{ int }
	t := SliceOf(TypeOf(T{}))
	ptrToThis := ValueOf(t).Elem().FieldByName("PtrToThis")
	if !ptrToThis.IsValid() {
		b.Skipf("%v has no ptrToThis field; was it removed from rtype?", t) // TODO fix this at top of refactoring
		// b.Fatalf("%v has no ptrToThis field; was it removed from rtype?", t)
	}
	if ptrToThis.Int() != 0 {
		b.Fatalf("%v.ptrToThis unexpectedly nonzero", t)
	}
	b.ResetTimer()

	// Now benchmark calling PointerTo on it: we'll have to hit the ptrMap cache on
	// every call.
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			PointerTo(t)
		}
	})
}

type B1 struct {
	X int
	Y int
	Z int
}

func BenchmarkFieldByName1(b *testing.B) {
	t := TypeOf(B1{})
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			t.FieldByName("Z")
		}
	})
}

func BenchmarkFieldByName2(b *testing.B) {
	t := TypeOf(S3{})
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			t.FieldByName("B")
		}
	})
}

func BenchmarkFieldByName3(b *testing.B) {
	t := TypeOf(R0{})
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			t.FieldByName("X")
		}
	})
}

type S struct {
	i1 int64
	i2 int64
}

func BenchmarkInterfaceBig(b *testing.B) {
	v := ValueOf(S{})
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v.Interface()
		}
	})
	b.StopTimer()
}

func BenchmarkInterfaceSmall(b *testing.B) {
	v := ValueOf(int64(0))
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v.Interface()
		}
	})
}

func BenchmarkNew(b *testing.B) {
	v := TypeOf(XM{})
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			New(v)
		}
	})
}

func BenchmarkMap(b *testing.B) {
	type V *int
	type S string
	value := ValueOf((V)(nil))
	stringKeys := []string{}
	mapOfStrings := map[string]V{}
	uint64Keys := []uint64{}
	mapOfUint64s := map[uint64]V{}
	userStringKeys := []S{}
	mapOfUserStrings := map[S]V{}
	for i := 0; i < 100; i++ {
		stringKey := fmt.Sprintf("key%d", i)
		stringKeys = append(stringKeys, stringKey)
		mapOfStrings[stringKey] = nil

		uint64Key := uint64(i)
		uint64Keys = append(uint64Keys, uint64Key)
		mapOfUint64s[uint64Key] = nil

		userStringKey := S(fmt.Sprintf("key%d", i))
		userStringKeys = append(userStringKeys, userStringKey)
		mapOfUserStrings[userStringKey] = nil
	}

	tests := []struct {
		label          string
		m, keys, value Value
	}{
		{"StringKeys", ValueOf(mapOfStrings), ValueOf(stringKeys), value},
		{"Uint64Keys", ValueOf(mapOfUint64s), ValueOf(uint64Keys), value},
		{"UserStringKeys", ValueOf(mapOfUserStrings), ValueOf(userStringKeys), value},
	}

	for _, tt := range tests {
		b.Run(tt.label, func(b *testing.B) {
			b.Run("MapIndex", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					for j := tt.keys.Len() - 1; j >= 0; j-- {
						tt.m.MapIndex(tt.keys.Index(j))
					}
				}
			})
			b.Run("SetMapIndex", func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					for j := tt.keys.Len() - 1; j >= 0; j-- {
						tt.m.SetMapIndex(tt.keys.Index(j), tt.value)
					}
				}
			})
		})
	}
}

func BenchmarkMapIterNext(b *testing.B) {
	m := ValueOf(map[string]int{"a": 0, "b": 1, "c": 2, "d": 3})
	it := m.MapRange()
	for i := 0; i < b.N; i++ {
		for it.Next() {
		}
		it.Reset(m)
	}
}
