// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sql

import (
	"database/sql/driver"
	"fmt"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

var someTime = time.Unix(123, 0)
var answer int64 = 42

type (
	userDefined       float64
	userDefinedSlice  []int
	userDefinedString string
)

type conversionTest struct {
	s, d interface{} // source and destination

	// following are used if they're non-zero
	wantint    int64
	wantuint   uint64
	wantstr    string
	wantbytes  []byte
	wantraw    RawBytes
	wantf32    float32
	wantf64    float64
	wanttime   time.Time
	wantbool   bool // used if d is of type *bool
	wanterr    string
	wantiface  interface{}
	wantptr    *int64 // if non-nil, *d's pointed value must be equal to *wantptr
	wantnil    bool   // if true, *d must be *int64(nil)
	wantusrdef userDefined
	wantusrstr userDefinedString
}

// Target variables for scanning into.
var (
	scanstr    string
	scanbytes  []byte
	scanraw    RawBytes
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
	scanptr    *int64
	scaniface  interface{}
)

func conversionTests() []conversionTest {
	// Return a fresh instance to test so "go test -count 2" works correctly.
	return []conversionTest{
		// Exact conversions (destination pointer type matches source type)
		{s: "foo", d: &scanstr, wantstr: "foo"},
		{s: 123, d: &scanint, wantint: 123},
		{s: someTime, d: &scantime, wanttime: someTime},

		// To strings
		{s: "string", d: &scanstr, wantstr: "string"},
		{s: []byte("byteslice"), d: &scanstr, wantstr: "byteslice"},
		{s: 123, d: &scanstr, wantstr: "123"},
		{s: int8(123), d: &scanstr, wantstr: "123"},
		{s: int64(123), d: &scanstr, wantstr: "123"},
		{s: uint8(123), d: &scanstr, wantstr: "123"},
		{s: uint16(123), d: &scanstr, wantstr: "123"},
		{s: uint32(123), d: &scanstr, wantstr: "123"},
		{s: uint64(123), d: &scanstr, wantstr: "123"},
		{s: 1.5, d: &scanstr, wantstr: "1.5"},

		// From time.Time:
		{s: time.Unix(1, 0).UTC(), d: &scanstr, wantstr: "1970-01-01T00:00:01Z"},
		{s: time.Unix(1453874597, 0).In(time.FixedZone("here", -3600*8)), d: &scanstr, wantstr: "2016-01-26T22:03:17-08:00"},
		{s: time.Unix(1, 2).UTC(), d: &scanstr, wantstr: "1970-01-01T00:00:01.000000002Z"},
		{s: time.Time{}, d: &scanstr, wantstr: "0001-01-01T00:00:00Z"},
		{s: time.Unix(1, 2).UTC(), d: &scanbytes, wantbytes: []byte("1970-01-01T00:00:01.000000002Z")},
		{s: time.Unix(1, 2).UTC(), d: &scaniface, wantiface: time.Unix(1, 2).UTC()},

		// To []byte
		{s: nil, d: &scanbytes, wantbytes: nil},
		{s: "string", d: &scanbytes, wantbytes: []byte("string")},
		{s: []byte("byteslice"), d: &scanbytes, wantbytes: []byte("byteslice")},
		{s: 123, d: &scanbytes, wantbytes: []byte("123")},
		{s: int8(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: int64(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: uint8(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: uint16(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: uint32(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: uint64(123), d: &scanbytes, wantbytes: []byte("123")},
		{s: 1.5, d: &scanbytes, wantbytes: []byte("1.5")},

		// To RawBytes
		{s: nil, d: &scanraw, wantraw: nil},
		{s: []byte("byteslice"), d: &scanraw, wantraw: RawBytes("byteslice")},
		{s: "string", d: &scanraw, wantraw: RawBytes("string")},
		{s: 123, d: &scanraw, wantraw: RawBytes("123")},
		{s: int8(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: int64(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: uint8(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: uint16(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: uint32(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: uint64(123), d: &scanraw, wantraw: RawBytes("123")},
		{s: 1.5, d: &scanraw, wantraw: RawBytes("1.5")},
		// time.Time has been placed here to check that the RawBytes slice gets
		// correctly reset when calling time.Time.AppendFormat.
		{s: time.Unix(2, 5).UTC(), d: &scanraw, wantraw: RawBytes("1970-01-01T00:00:02.000000005Z")},

		// Strings to integers
		{s: "255", d: &scanuint8, wantuint: 255},
		{s: "256", d: &scanuint8, wanterr: "converting driver.Value type string (\"256\") to a uint8: value out of range"},
		{s: "256", d: &scanuint16, wantuint: 256},
		{s: "-1", d: &scanint, wantint: -1},
		{s: "foo", d: &scanint, wanterr: "converting driver.Value type string (\"foo\") to a int: invalid syntax"},

		// int64 to smaller integers
		{s: int64(5), d: &scanuint8, wantuint: 5},
		{s: int64(256), d: &scanuint8, wanterr: "converting driver.Value type int64 (\"256\") to a uint8: value out of range"},
		{s: int64(256), d: &scanuint16, wantuint: 256},
		{s: int64(65536), d: &scanuint16, wanterr: "converting driver.Value type int64 (\"65536\") to a uint16: value out of range"},

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

		// Pointers
		{s: interface{}(nil), d: &scanptr, wantnil: true},
		{s: int64(42), d: &scanptr, wantptr: &answer},

		// To interface{}
		{s: float64(1.5), d: &scaniface, wantiface: float64(1.5)},
		{s: int64(1), d: &scaniface, wantiface: int64(1)},
		{s: "str", d: &scaniface, wantiface: "str"},
		{s: []byte("byteslice"), d: &scaniface, wantiface: []byte("byteslice")},
		{s: true, d: &scaniface, wantiface: true},
		{s: nil, d: &scaniface},
		{s: []byte(nil), d: &scaniface, wantiface: []byte(nil)},

		// To a user-defined type
		{s: 1.5, d: new(userDefined), wantusrdef: 1.5},
		{s: int64(123), d: new(userDefined), wantusrdef: 123},
		{s: "1.5", d: new(userDefined), wantusrdef: 1.5},
		{s: []byte{1, 2, 3}, d: new(userDefinedSlice), wanterr: `unsupported Scan, storing driver.Value type []uint8 into type *sql.userDefinedSlice`},
		{s: "str", d: new(userDefinedString), wantusrstr: "str"},

		// Other errors
		{s: complex(1, 2), d: &scanstr, wanterr: `unsupported Scan, storing driver.Value type complex128 into type *string`},
	}
}

func intPtrValue(intptr interface{}) interface{} {
	return reflect.Indirect(reflect.Indirect(reflect.ValueOf(intptr))).Int()
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
	for n, ct := range conversionTests() {
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
		if ct.wantbytes != nil && string(ct.wantbytes) != string(scanbytes) {
			errf("want byte %q, got %q", ct.wantbytes, scanbytes)
		}
		if ct.wantraw != nil && string(ct.wantraw) != string(scanraw) {
			errf("want RawBytes %q, got %q", ct.wantraw, scanraw)
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
		if ct.wantnil && *ct.d.(**int64) != nil {
			errf("want nil, got %v", intPtrValue(ct.d))
		}
		if ct.wantptr != nil {
			if *ct.d.(**int64) == nil {
				errf("want pointer to %v, got nil", *ct.wantptr)
			} else if *ct.wantptr != intPtrValue(ct.d) {
				errf("want pointer to %v, got %v", *ct.wantptr, intPtrValue(ct.d))
			}
		}
		if ifptr, ok := ct.d.(*interface{}); ok {
			if !reflect.DeepEqual(ct.wantiface, scaniface) {
				errf("want interface %#v, got %#v", ct.wantiface, scaniface)
				continue
			}
			if srcBytes, ok := ct.s.([]byte); ok {
				dstBytes := (*ifptr).([]byte)
				if len(srcBytes) > 0 && &dstBytes[0] == &srcBytes[0] {
					errf("copy into interface{} didn't copy []byte data")
				}
			}
		}
		if ct.wantusrdef != 0 && ct.wantusrdef != *ct.d.(*userDefined) {
			errf("want userDefined %f, got %f", ct.wantusrdef, *ct.d.(*userDefined))
		}
		if len(ct.wantusrstr) != 0 && ct.wantusrstr != *ct.d.(*userDefinedString) {
			errf("want userDefined %q, got %q", ct.wantusrstr, *ct.d.(*userDefinedString))
		}
	}
}

func TestNullString(t *testing.T) {
	var ns NullString
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

type valueConverterTest struct {
	c       driver.ValueConverter
	in, out interface{}
	err     string
}

var valueConverterTests = []valueConverterTest{
	{driver.DefaultParameterConverter, NullString{"hi", true}, "hi", ""},
	{driver.DefaultParameterConverter, NullString{"", false}, nil, ""},
}

func TestValueConverters(t *testing.T) {
	for i, tt := range valueConverterTests {
		out, err := tt.c.ConvertValue(tt.in)
		goterr := ""
		if err != nil {
			goterr = err.Error()
		}
		if goterr != tt.err {
			t.Errorf("test %d: %T(%T(%v)) error = %q; want error = %q",
				i, tt.c, tt.in, tt.in, goterr, tt.err)
		}
		if tt.err != "" {
			continue
		}
		if !reflect.DeepEqual(out, tt.out) {
			t.Errorf("test %d: %T(%T(%v)) = %v (%T); want %v (%T)",
				i, tt.c, tt.in, tt.in, out, out, tt.out, tt.out)
		}
	}
}

// Tests that assigning to RawBytes doesn't allocate (and also works).
func TestRawBytesAllocs(t *testing.T) {
	var tests = []struct {
		name string
		in   interface{}
		want string
	}{
		{"uint64", uint64(12345678), "12345678"},
		{"uint32", uint32(1234), "1234"},
		{"uint16", uint16(12), "12"},
		{"uint8", uint8(1), "1"},
		{"uint", uint(123), "123"},
		{"int", int(123), "123"},
		{"int8", int8(1), "1"},
		{"int16", int16(12), "12"},
		{"int32", int32(1234), "1234"},
		{"int64", int64(12345678), "12345678"},
		{"float32", float32(1.5), "1.5"},
		{"float64", float64(64), "64"},
		{"bool", false, "false"},
		{"time", time.Unix(2, 5).UTC(), "1970-01-01T00:00:02.000000005Z"},
	}

	buf := make(RawBytes, 10)
	test := func(name string, in interface{}, want string) {
		if err := convertAssign(&buf, in); err != nil {
			t.Fatalf("%s: convertAssign = %v", name, err)
		}
		match := len(buf) == len(want)
		if match {
			for i, b := range buf {
				if want[i] != b {
					match = false
					break
				}
			}
		}
		if !match {
			t.Fatalf("%s: got %q (len %d); want %q (len %d)", name, buf, len(buf), want, len(want))
		}
	}

	n := testing.AllocsPerRun(100, func() {
		for _, tt := range tests {
			test(tt.name, tt.in, tt.want)
		}
	})

	// The numbers below are only valid for 64-bit interface word sizes,
	// and gc. With 32-bit words there are more convT2E allocs, and
	// with gccgo, only pointers currently go in interface data.
	// So only care on amd64 gc for now.
	measureAllocs := runtime.GOARCH == "amd64" && runtime.Compiler == "gc"

	if n > 0.5 && measureAllocs {
		t.Fatalf("allocs = %v; want 0", n)
	}

	// This one involves a convT2E allocation, string -> interface{}
	n = testing.AllocsPerRun(100, func() {
		test("string", "foo", "foo")
	})
	if n > 1.5 && measureAllocs {
		t.Fatalf("allocs = %v; want max 1", n)
	}
}

// https://golang.org/issues/13905
func TestUserDefinedBytes(t *testing.T) {
	type userDefinedBytes []byte
	var u userDefinedBytes
	v := []byte("foo")

	convertAssign(&u, v)
	if &u[0] == &v[0] {
		t.Fatal("userDefinedBytes got potentially dirty driver memory")
	}
}

type Valuer_V string

func (v Valuer_V) Value() (driver.Value, error) {
	return strings.ToUpper(string(v)), nil
}

type Valuer_P string

func (p *Valuer_P) Value() (driver.Value, error) {
	if p == nil {
		return "nil-to-str", nil
	}
	return strings.ToUpper(string(*p)), nil
}

func TestDriverArgs(t *testing.T) {
	var nilValuerVPtr *Valuer_V
	var nilValuerPPtr *Valuer_P
	var nilStrPtr *string
	tests := []struct {
		args []interface{}
		want []driver.NamedValue
	}{
		0: {
			args: []interface{}{Valuer_V("foo")},
			want: []driver.NamedValue{
				{
					Ordinal: 1,
					Value:   "FOO",
				},
			},
		},
		1: {
			args: []interface{}{nilValuerVPtr},
			want: []driver.NamedValue{
				{
					Ordinal: 1,
					Value:   nil,
				},
			},
		},
		2: {
			args: []interface{}{nilValuerPPtr},
			want: []driver.NamedValue{
				{
					Ordinal: 1,
					Value:   "nil-to-str",
				},
			},
		},
		3: {
			args: []interface{}{"plain-str"},
			want: []driver.NamedValue{
				{
					Ordinal: 1,
					Value:   "plain-str",
				},
			},
		},
		4: {
			args: []interface{}{nilStrPtr},
			want: []driver.NamedValue{
				{
					Ordinal: 1,
					Value:   nil,
				},
			},
		},
	}
	for i, tt := range tests {
		ds := &driverStmt{Locker: &sync.Mutex{}, si: stubDriverStmt{nil}}
		got, err := driverArgsConnLocked(nil, ds, tt.args)
		if err != nil {
			t.Errorf("test[%d]: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("test[%d]: got %v, want %v", i, got, tt.want)
		}
	}
}

type dec struct {
	form        byte
	neg         bool
	coefficient [16]byte
	exponent    int32
}

func (d dec) Decompose(buf []byte) (form byte, negative bool, coefficient []byte, exponent int32) {
	coef := make([]byte, 16)
	copy(coef, d.coefficient[:])
	return d.form, d.neg, coef, d.exponent
}

func (d *dec) Compose(form byte, negative bool, coefficient []byte, exponent int32) error {
	switch form {
	default:
		return fmt.Errorf("unknown form %d", form)
	case 1, 2:
		d.form = form
		d.neg = negative
		return nil
	case 0:
	}
	d.form = form
	d.neg = negative
	d.exponent = exponent

	// This isn't strictly correct, as the extra bytes could be all zero,
	// ignore this for this test.
	if len(coefficient) > 16 {
		return fmt.Errorf("coefficient too large")
	}
	copy(d.coefficient[:], coefficient)

	return nil
}

type decFinite struct {
	neg         bool
	coefficient [16]byte
	exponent    int32
}

func (d decFinite) Decompose(buf []byte) (form byte, negative bool, coefficient []byte, exponent int32) {
	coef := make([]byte, 16)
	copy(coef, d.coefficient[:])
	return 0, d.neg, coef, d.exponent
}

func (d *decFinite) Compose(form byte, negative bool, coefficient []byte, exponent int32) error {
	switch form {
	default:
		return fmt.Errorf("unknown form %d", form)
	case 1, 2:
		return fmt.Errorf("unsupported form %d", form)
	case 0:
	}
	d.neg = negative
	d.exponent = exponent

	// This isn't strictly correct, as the extra bytes could be all zero,
	// ignore this for this test.
	if len(coefficient) > 16 {
		return fmt.Errorf("coefficient too large")
	}
	copy(d.coefficient[:], coefficient)

	return nil
}

func TestDecimal(t *testing.T) {
	list := []struct {
		name string
		in   decimalDecompose
		out  dec
		err  bool
	}{
		{name: "same", in: dec{exponent: -6}, out: dec{exponent: -6}},

		// Ensure reflection is not used to assign the value by using different types.
		{name: "diff", in: decFinite{exponent: -6}, out: dec{exponent: -6}},

		{name: "bad-form", in: dec{form: 200}, err: true},
	}
	for _, item := range list {
		t.Run(item.name, func(t *testing.T) {
			out := dec{}
			err := convertAssign(&out, item.in)
			if item.err {
				if err == nil {
					t.Fatalf("unexpected nil error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(out, item.out) {
				t.Fatalf("got %#v want %#v", out, item.out)
			}
		})
	}
}
