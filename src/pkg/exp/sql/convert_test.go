// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"fmt"
	"reflect"
	"testing"
	"time"
)

var someTime = time.Unix(123, 0)

type conversionTest struct {
	s, d interface{} // source and destination

	// following are used if they're non-zero
	wantint  int64
	wantuint uint64
	wantstr  string
	wantf32  float32
	wantf64  float64
	wanttime time.Time
	wantbool bool // used if d is of type *bool
	wanterr  string
}

// Target variables for scanning into.
var (
	scanstr    string
	scanint    int
	scanint8   int8
	scanint16  int16
	scanint32  int32
	scanuint8  uint8
	scanuint16 uint16
	scanbool   bool
	scanf32    float32
	scanf64    float64
	scantime   time.Time
)

var conversionTests = []conversionTest{
	// Exact conversions (destination pointer type matches source type)
	{s: "foo", d: &scanstr, wantstr: "foo"},
	{s: 123, d: &scanint, wantint: 123},
	{s: someTime, d: &scantime, wanttime: someTime},

	// To strings
	{s: []byte("byteslice"), d: &scanstr, wantstr: "byteslice"},
	{s: 123, d: &scanstr, wantstr: "123"},
	{s: int8(123), d: &scanstr, wantstr: "123"},
	{s: int64(123), d: &scanstr, wantstr: "123"},
	{s: uint8(123), d: &scanstr, wantstr: "123"},
	{s: uint16(123), d: &scanstr, wantstr: "123"},
	{s: uint32(123), d: &scanstr, wantstr: "123"},
	{s: uint64(123), d: &scanstr, wantstr: "123"},
	{s: 1.5, d: &scanstr, wantstr: "1.5"},

	// Strings to integers
	{s: "255", d: &scanuint8, wantuint: 255},
	{s: "256", d: &scanuint8, wanterr: `converting string "256" to a uint8: strconv.ParseUint: parsing "256": value out of range`},
	{s: "256", d: &scanuint16, wantuint: 256},
	{s: "-1", d: &scanint, wantint: -1},
	{s: "foo", d: &scanint, wanterr: `converting string "foo" to a int: strconv.ParseInt: parsing "foo": invalid syntax`},

	// True bools
	{s: true, d: &scanbool, wantbool: true},
	{s: "True", d: &scanbool, wantbool: true},
	{s: "TRUE", d: &scanbool, wantbool: true},
	{s: "1", d: &scanbool, wantbool: true},
	{s: 1, d: &scanbool, wantbool: true},
	{s: int64(1), d: &scanbool, wantbool: true},
	{s: uint16(1), d: &scanbool, wantbool: true},

	// False bools
	{s: false, d: &scanbool, wantbool: false},
	{s: "false", d: &scanbool, wantbool: false},
	{s: "FALSE", d: &scanbool, wantbool: false},
	{s: "0", d: &scanbool, wantbool: false},
	{s: 0, d: &scanbool, wantbool: false},
	{s: int64(0), d: &scanbool, wantbool: false},
	{s: uint16(0), d: &scanbool, wantbool: false},

	// Not bools
	{s: "yup", d: &scanbool, wanterr: `sql/driver: couldn't convert "yup" into type bool`},
	{s: 2, d: &scanbool, wanterr: `sql/driver: couldn't convert 2 into type bool`},

	// Floats
	{s: float64(1.5), d: &scanf64, wantf64: float64(1.5)},
	{s: int64(1), d: &scanf64, wantf64: float64(1)},
	{s: float64(1.5), d: &scanf32, wantf32: float32(1.5)},
	{s: "1.5", d: &scanf32, wantf32: float32(1.5)},
	{s: "1.5", d: &scanf64, wantf64: float64(1.5)},
}

func intValue(intptr interface{}) int64 {
	return reflect.Indirect(reflect.ValueOf(intptr)).Int()
}

func uintValue(intptr interface{}) uint64 {
	return reflect.Indirect(reflect.ValueOf(intptr)).Uint()
}

func float64Value(ptr interface{}) float64 {
	return *(ptr.(*float64))
}

func float32Value(ptr interface{}) float32 {
	return *(ptr.(*float32))
}

func timeValue(ptr interface{}) time.Time {
	return *(ptr.(*time.Time))
}

func TestConversions(t *testing.T) {
	for n, ct := range conversionTests {
		err := convertAssign(ct.d, ct.s)
		errstr := ""
		if err != nil {
			errstr = err.Error()
		}
		errf := func(format string, args ...interface{}) {
			base := fmt.Sprintf("convertAssign #%d: for %v (%T) -> %T, ", n, ct.s, ct.s, ct.d)
			t.Errorf(base+format, args...)
		}
		if errstr != ct.wanterr {
			errf("got error %q, want error %q", errstr, ct.wanterr)
		}
		if ct.wantstr != "" && ct.wantstr != scanstr {
			errf("want string %q, got %q", ct.wantstr, scanstr)
		}
		if ct.wantint != 0 && ct.wantint != intValue(ct.d) {
			errf("want int %d, got %d", ct.wantint, intValue(ct.d))
		}
		if ct.wantuint != 0 && ct.wantuint != uintValue(ct.d) {
			errf("want uint %d, got %d", ct.wantuint, uintValue(ct.d))
		}
		if ct.wantf32 != 0 && ct.wantf32 != float32Value(ct.d) {
			errf("want float32 %v, got %v", ct.wantf32, float32Value(ct.d))
		}
		if ct.wantf64 != 0 && ct.wantf64 != float64Value(ct.d) {
			errf("want float32 %v, got %v", ct.wantf64, float64Value(ct.d))
		}
		if bp, boolTest := ct.d.(*bool); boolTest && *bp != ct.wantbool && ct.wanterr == "" {
			errf("want bool %v, got %v", ct.wantbool, *bp)
		}
		if !ct.wanttime.IsZero() && !ct.wanttime.Equal(timeValue(ct.d)) {
			errf("want time %v, got %v", ct.wanttime, timeValue(ct.d))
		}
	}
}

func TestNullableString(t *testing.T) {
	var ns NullableString
	convertAssign(&ns, []byte("foo"))
	if !ns.Valid {
		t.Errorf("expecting not null")
	}
	if ns.String != "foo" {
		t.Errorf("expecting foo; got %q", ns.String)
	}
	convertAssign(&ns, nil)
	if ns.Valid {
		t.Errorf("expecting null on nil")
	}
	if ns.String != "" {
		t.Errorf("expecting blank on nil; got %q", ns.String)
	}
}
