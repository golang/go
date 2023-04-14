// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package slog

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"
	"unsafe"
)

func TestValueEqual(t *testing.T) {
	var x, y int
	vals := []Value{
		{},
		Int64Value(1),
		Int64Value(2),
		Float64Value(3.5),
		Float64Value(3.7),
		BoolValue(true),
		BoolValue(false),
		TimeValue(testTime),
		AnyValue(&x),
		AnyValue(&y),
		GroupValue(Bool("b", true), Int("i", 3)),
	}
	for i, v1 := range vals {
		for j, v2 := range vals {
			got := v1.Equal(v2)
			want := i == j
			if got != want {
				t.Errorf("%v.Equal(%v): got %t, want %t", v1, v2, got, want)
			}
		}
	}
}

func panics(f func()) (b bool) {
	defer func() {
		if x := recover(); x != nil {
			b = true
		}
	}()
	f()
	return false
}

func TestValueString(t *testing.T) {
	for _, test := range []struct {
		v    Value
		want string
	}{
		{Int64Value(-3), "-3"},
		{Float64Value(.15), "0.15"},
		{BoolValue(true), "true"},
		{StringValue("foo"), "foo"},
		{TimeValue(testTime), "2000-01-02 03:04:05 +0000 UTC"},
		{AnyValue(time.Duration(3 * time.Second)), "3s"},
		{GroupValue(Int("a", 1), Bool("b", true)), "[a=1 b=true]"},
	} {
		if got := test.v.String(); got != test.want {
			t.Errorf("%#v:\ngot  %q\nwant %q", test.v, got, test.want)
		}
	}
}

func TestValueNoAlloc(t *testing.T) {
	// Assign values just to make sure the compiler doesn't optimize away the statements.
	var (
		i  int64
		u  uint64
		f  float64
		b  bool
		s  string
		x  any
		p  = &i
		d  time.Duration
		tm time.Time
	)
	a := int(testing.AllocsPerRun(5, func() {
		i = Int64Value(1).Int64()
		u = Uint64Value(1).Uint64()
		f = Float64Value(1).Float64()
		b = BoolValue(true).Bool()
		s = StringValue("foo").String()
		d = DurationValue(d).Duration()
		tm = TimeValue(testTime).Time()
		x = AnyValue(p).Any()
	}))
	if a != 0 {
		t.Errorf("got %d allocs, want zero", a)
	}
	_ = u
	_ = f
	_ = b
	_ = s
	_ = x
	_ = tm
}

func TestAnyLevelAlloc(t *testing.T) {
	// Because typical Levels are small integers,
	// they are zero-alloc.
	var a Value
	x := LevelDebug + 100
	wantAllocs(t, 0, func() { a = AnyValue(x) })
	_ = a
}

func TestAnyValue(t *testing.T) {
	for _, test := range []struct {
		in   any
		want Value
	}{
		{1, IntValue(1)},
		{1.5, Float64Value(1.5)},
		{"s", StringValue("s")},
		{uint(2), Uint64Value(2)},
		{true, BoolValue(true)},
		{testTime, TimeValue(testTime)},
		{time.Hour, DurationValue(time.Hour)},
		{[]Attr{Int("i", 3)}, GroupValue(Int("i", 3))},
		{IntValue(4), IntValue(4)},
	} {
		got := AnyValue(test.in)
		if !got.Equal(test.want) {
			t.Errorf("%v (%[1]T): got %v (kind %s), want %v (kind %s)",
				test.in, got, got.Kind(), test.want, test.want.Kind())
		}
	}
}

func TestValueAny(t *testing.T) {
	for _, want := range []any{
		nil,
		LevelDebug + 100,
		time.UTC, // time.Locations treated specially...
		KindBool, // ...as are Kinds
		[]Attr{Int("a", 1)},
	} {
		v := AnyValue(want)
		got := v.Any()
		if !reflect.DeepEqual(got, want) {
			t.Errorf("got %v, want %v", got, want)
		}
	}
}

func TestLogValue(t *testing.T) {
	want := "replaced"
	r := &replace{StringValue(want)}
	v := AnyValue(r)
	if g, w := v.Kind(), KindLogValuer; g != w {
		t.Errorf("got %s, want %s", g, w)
	}
	got := v.LogValuer().LogValue().Any()
	if got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}

	// Test Resolve.
	got = v.Resolve().Any()
	if got != want {
		t.Errorf("got %#v, want %#v", got, want)
	}

	// Test Resolve max iteration.
	r.v = AnyValue(r) // create a cycle
	got = AnyValue(r).Resolve().Any()
	if _, ok := got.(error); !ok {
		t.Errorf("expected error, got %T", got)
	}

	// Groups are not recursively resolved.
	c := Any("c", &replace{StringValue("d")})
	v = AnyValue(&replace{GroupValue(Int("a", 1), Group("b", c))})
	got2 := v.Resolve().Any().([]Attr)
	want2 := []Attr{Int("a", 1), Group("b", c)}
	if !attrsEqual(got2, want2) {
		t.Errorf("got %v, want %v", got2, want2)
	}

	// Verify that panics in Resolve are caught and turn into errors.
	v = AnyValue(panickingLogValue{})
	got = v.Resolve().Any()
	gotErr, ok := got.(error)
	if !ok {
		t.Errorf("expected error, got %T", got)
	}
	// The error should provide some context information.
	// We'll just check that this function name appears in it.
	if got, want := gotErr.Error(), "TestLogValue"; !strings.Contains(got, want) {
		t.Errorf("got %q, want substring %q", got, want)
	}
}

func TestZeroTime(t *testing.T) {
	z := time.Time{}
	got := TimeValue(z).Time()
	if !got.IsZero() {
		t.Errorf("got %s (%#[1]v), not zero time (%#v)", got, z)
	}
}

type replace struct {
	v Value
}

func (r *replace) LogValue() Value { return r.v }

type panickingLogValue struct{}

func (panickingLogValue) LogValue() Value { panic("bad") }

// A Value with "unsafe" strings is significantly faster:
// safe:  1785 ns/op, 0 allocs
// unsafe: 690 ns/op, 0 allocs

// Run this with and without -tags unsafe_kvs to compare.
func BenchmarkUnsafeStrings(b *testing.B) {
	b.ReportAllocs()
	dst := make([]Value, 100)
	src := make([]Value, len(dst))
	b.Logf("Value size = %d", unsafe.Sizeof(Value{}))
	for i := range src {
		src[i] = StringValue(fmt.Sprintf("string#%d", i))
	}
	b.ResetTimer()
	var d string
	for i := 0; i < b.N; i++ {
		copy(dst, src)
		for _, a := range dst {
			d = a.String()
		}
	}
	_ = d
}
