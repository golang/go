// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package json

import (
	stdjson "encoding/json"
	"errors"
	"math"
	"math/rand/v2"
	"testing"
)

func floatTable() []float64 {
	return []float64{
		0, math.Copysign(0, -1), 1, -1, 0.5, 0.1, -0.1, 1e21, -1e21, 1e20, 1e-7, -1e-7, 1e-6, 9.999999e-7,
		1e-5, 123456789, 5e-324, -5e-324, math.MaxFloat64, -math.MaxFloat64, math.SmallestNonzeroFloat32,
		math.MaxFloat32, 16777216, 16777217, 100, 1e6, 12345.678, math.Pi, 1e300, 1e-300, 1e100, 1e-100,
		9007199254740993, 1.0000000000000002e-6, 0.000001234567890123456, 123456789012345680000,
		2.2250738585072014e-308, 1e22, 1.5e-9, 1.5e-10, 2e-9, 1e-10,
	}
}

func TestAppendFloatParity64(t *testing.T) {
	for _, f := range floatTable() {
		want, err := stdjson.Marshal(f)
		if err != nil {
			t.Fatal(err)
		}
		if got := AppendFloat(nil, f, 64); string(got) != string(want) {
			t.Errorf("AppendFloat(%v, 64) = %s, want %s", f, got, want)
		}
	}
}

func TestAppendFloatParity32(t *testing.T) {
	for _, f := range floatTable() {
		f32 := float32(f)
		if math.IsInf(float64(f32), 0) {
			continue
		}
		want, err := stdjson.Marshal(f32)
		if err != nil {
			t.Fatal(err)
		}
		if got := AppendFloat(nil, float64(f32), 32); string(got) != string(want) {
			t.Errorf("AppendFloat(%v, 32) = %s, want %s", f32, got, want)
		}
	}
}

func TestAppendFloatKeepsPrefix(t *testing.T) {
	if got := string(AppendFloat([]byte("x="), 1.5e-9, 64)); got != "x=1.5e-9" {
		t.Errorf("AppendFloat = %q", got)
	}
}

func TestAppendFloatSweepParity(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	for range 20000 {
		f := math.Float64frombits(r.Uint64())
		if math.IsNaN(f) || math.IsInf(f, 0) {
			continue
		}
		want, _ := stdjson.Marshal(f)
		if got := AppendFloat(nil, f, 64); string(got) != string(want) {
			t.Fatalf("AppendFloat(%v, 64) = %s, want %s", f, got, want)
		}
		g := math.Float32frombits(r.Uint32())
		if math.IsNaN(float64(g)) || math.IsInf(float64(g), 0) {
			continue
		}
		want, _ = stdjson.Marshal(g)
		if got := AppendFloat(nil, float64(g), 32); string(got) != string(want) {
			t.Fatalf("AppendFloat(%v, 32) = %s, want %s", g, got, want)
		}
	}
}

func TestFloatErr(t *testing.T) {
	if err := FloatErr(1.5, 1.5, 64); err != nil {
		t.Errorf("FloatErr(1.5) = %v", err)
	}
	type myFloat float32
	cases := []struct {
		v    any
		f    float64
		bits int
	}{
		{math.NaN(), math.NaN(), 64},
		{math.Inf(1), math.Inf(1), 64},
		{math.Inf(-1), math.Inf(-1), 64},
		{float32(math.Inf(1)), math.Inf(1), 32},
		{myFloat(math.NaN()), math.NaN(), 32},
	}
	for _, c := range cases {
		_, want := stdjson.Marshal(c.v)
		got := FloatErr(c.v, c.f, c.bits)
		if want == nil || got == nil {
			t.Fatalf("FloatErr(%v) = %v, std = %v", c.v, got, want)
		}
		if got.Error() != want.Error() {
			t.Errorf("FloatErr(%v) = %q, want %q", c.v, got, want)
		}
		if _, ok := errors.AsType[*stdjson.UnsupportedValueError](got); !ok {
			t.Errorf("FloatErr(%v) = %T, want *UnsupportedValueError", c.v, got)
		}
	}
}

func finite64(f float64) bool {
	return !math.IsNaN(f) && !math.IsInf(f, 0)
}

func widest64(r *rand.Rand) int {
	extremes := append(floatTable(),
		-1.0000000000000002e-06, -math.Nextafter(1e-6, 1), -math.Nextafter(1e21, 0), -math.Nextafter(1e-6, 0),
		-math.Nextafter(1e21, math.Inf(1)), -math.Nextafter(1e-5, 0), -2.2250738585072014e-308,
	)
	widest := 0
	for _, f := range extremes {
		widest = max(widest, len(AppendFloat(nil, f, 64)), len(AppendFloat(nil, -f, 64)))
	}
	for range 300000 {
		if f := math.Float64frombits(r.Uint64()); finite64(f) {
			widest = max(widest, len(AppendFloat(nil, f, 64)))
		}
	}
	for range 300000 {
		f := math.Float64frombits(math.Float64bits(1e-6) + r.Uint64()%4096)
		g := 1e-6 * (1 + r.Float64()*8)
		widest = max(widest, len(AppendFloat(nil, -f, 64)), len(AppendFloat(nil, -g, 64)))
	}
	return widest
}

func widest32(r *rand.Rand) int {
	extremes := []float32{
		math.MaxFloat32, math.SmallestNonzeroFloat32, 1e20, 1e21, 1e-6, 1e-7, 16777216, 1.2345679e20,
		math.Nextafter32(1e21, 0), math.Nextafter32(1e-6, 1), math.Nextafter32(1e-6, 0), 0.1, 123456790000000000000,
	}
	widest := 0
	for _, f := range extremes {
		widest = max(widest, len(AppendFloat(nil, float64(f), 32)), len(AppendFloat(nil, -float64(f), 32)))
	}
	for range 300000 {
		if f := float64(math.Float32frombits(r.Uint32())); finite64(f) {
			widest = max(widest, len(AppendFloat(nil, f, 32)))
		}
	}
	for range 300000 {
		f := math.Float32frombits(math.Float32bits(1e20) + r.Uint32()%(1<<24))
		if f < 1e21 {
			widest = max(widest, len(AppendFloat(nil, -float64(f), 32)))
		}
	}
	return widest
}

func checkTightBound(t *testing.T, name string, observed, bound int) {
	t.Helper()
	if observed > bound {
		t.Fatalf("observed %d bytes, %s = %d", observed, name, bound)
	}
	if observed != bound {
		t.Errorf("%s = %d is not tight, observed maximum %d", name, bound, observed)
	}
}

func TestMaxFloatLen(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	checkTightBound(t, "MaxFloatLen64", widest64(r), MaxFloatLen64)
	checkTightBound(t, "MaxFloatLen32", widest32(r), MaxFloatLen32)
}
