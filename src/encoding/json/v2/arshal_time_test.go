// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"fmt"
	"math"
	"testing"
	"time"

	"encoding/json/internal/jsonwire"
)

func baseLabel(base uint64) string {
	if log10 := math.Log10(float64(base)); log10 == float64(int64(log10)) {
		return fmt.Sprintf("1e%d", int(log10))
	}
	return fmt.Sprint(base)
}

var formatDurationTestdata = []struct {
	td          time.Duration
	base10Sec   string
	base10Milli string
	base10Micro string
	base10Nano  string
}{
	{math.MaxInt64, "9223372036.854775807", "9223372036854.775807", "9223372036854775.807", "9223372036854775807"},
	{1e12 + 1e12, "2000", "2000000", "2000000000", "2000000000000"},
	{1e12 + 1e11, "1100", "1100000", "1100000000", "1100000000000"},
	{1e12 + 1e10, "1010", "1010000", "1010000000", "1010000000000"},
	{1e12 + 1e9, "1001", "1001000", "1001000000", "1001000000000"},
	{1e12 + 1e8, "1000.1", "1000100", "1000100000", "1000100000000"},
	{1e12 + 1e7, "1000.01", "1000010", "1000010000", "1000010000000"},
	{1e12 + 1e6, "1000.001", "1000001", "1000001000", "1000001000000"},
	{1e12 + 1e5, "1000.0001", "1000000.1", "1000000100", "1000000100000"},
	{1e12 + 1e4, "1000.00001", "1000000.01", "1000000010", "1000000010000"},
	{1e12 + 1e3, "1000.000001", "1000000.001", "1000000001", "1000000001000"},
	{1e12 + 1e2, "1000.0000001", "1000000.0001", "1000000000.1", "1000000000100"},
	{1e12 + 1e1, "1000.00000001", "1000000.00001", "1000000000.01", "1000000000010"},
	{1e12 + 1e0, "1000.000000001", "1000000.000001", "1000000000.001", "1000000000001"},
	{+(1e9 + 1), "1.000000001", "1000.000001", "1000000.001", "1000000001"},
	{+(1e9), "1", "1000", "1000000", "1000000000"},
	{+(1e9 - 1), "0.999999999", "999.999999", "999999.999", "999999999"},
	{+100000000, "0.1", "100", "100000", "100000000"},
	{+120000000, "0.12", "120", "120000", "120000000"},
	{+123000000, "0.123", "123", "123000", "123000000"},
	{+123400000, "0.1234", "123.4", "123400", "123400000"},
	{+123450000, "0.12345", "123.45", "123450", "123450000"},
	{+123456000, "0.123456", "123.456", "123456", "123456000"},
	{+123456700, "0.1234567", "123.4567", "123456.7", "123456700"},
	{+123456780, "0.12345678", "123.45678", "123456.78", "123456780"},
	{+123456789, "0.123456789", "123.456789", "123456.789", "123456789"},
	{+12345678, "0.012345678", "12.345678", "12345.678", "12345678"},
	{+1234567, "0.001234567", "1.234567", "1234.567", "1234567"},
	{+123456, "0.000123456", "0.123456", "123.456", "123456"},
	{+12345, "0.000012345", "0.012345", "12.345", "12345"},
	{+1234, "0.000001234", "0.001234", "1.234", "1234"},
	{+123, "0.000000123", "0.000123", "0.123", "123"},
	{+12, "0.000000012", "0.000012", "0.012", "12"},
	{+1, "0.000000001", "0.000001", "0.001", "1"},
	{0, "0", "0", "0", "0"},
	{-1, "-0.000000001", "-0.000001", "-0.001", "-1"},
	{-12, "-0.000000012", "-0.000012", "-0.012", "-12"},
	{-123, "-0.000000123", "-0.000123", "-0.123", "-123"},
	{-1234, "-0.000001234", "-0.001234", "-1.234", "-1234"},
	{-12345, "-0.000012345", "-0.012345", "-12.345", "-12345"},
	{-123456, "-0.000123456", "-0.123456", "-123.456", "-123456"},
	{-1234567, "-0.001234567", "-1.234567", "-1234.567", "-1234567"},
	{-12345678, "-0.012345678", "-12.345678", "-12345.678", "-12345678"},
	{-123456789, "-0.123456789", "-123.456789", "-123456.789", "-123456789"},
	{-123456780, "-0.12345678", "-123.45678", "-123456.78", "-123456780"},
	{-123456700, "-0.1234567", "-123.4567", "-123456.7", "-123456700"},
	{-123456000, "-0.123456", "-123.456", "-123456", "-123456000"},
	{-123450000, "-0.12345", "-123.45", "-123450", "-123450000"},
	{-123400000, "-0.1234", "-123.4", "-123400", "-123400000"},
	{-123000000, "-0.123", "-123", "-123000", "-123000000"},
	{-120000000, "-0.12", "-120", "-120000", "-120000000"},
	{-100000000, "-0.1", "-100", "-100000", "-100000000"},
	{-(1e9 - 1), "-0.999999999", "-999.999999", "-999999.999", "-999999999"},
	{-(1e9), "-1", "-1000", "-1000000", "-1000000000"},
	{-(1e9 + 1), "-1.000000001", "-1000.000001", "-1000000.001", "-1000000001"},
	{math.MinInt64, "-9223372036.854775808", "-9223372036854.775808", "-9223372036854775.808", "-9223372036854775808"},
}

func TestFormatDuration(t *testing.T) {
	var gotBuf []byte
	check := func(td time.Duration, s string, base uint64) {
		a := durationArshaler{td, base}
		gotBuf, _ = a.appendMarshal(gotBuf[:0])
		if string(gotBuf) != s {
			t.Errorf("formatDuration(%d, %s) = %q, want %q", td, baseLabel(base), string(gotBuf), s)
		}
		if err := a.unmarshal(gotBuf); err != nil {
			t.Errorf("parseDuration(%q, %s) error: %v", gotBuf, baseLabel(base), err)
		}
		if a.td != td {
			t.Errorf("parseDuration(%q, %s) = %d, want %d", gotBuf, baseLabel(base), a.td, td)
		}
	}
	for _, tt := range formatDurationTestdata {
		check(tt.td, tt.base10Sec, 1e9)
		check(tt.td, tt.base10Milli, 1e6)
		check(tt.td, tt.base10Micro, 1e3)
		check(tt.td, tt.base10Nano, 1e0)
	}
}

var parseDurationTestdata = []struct {
	in      string
	base    uint64
	want    time.Duration
	wantErr bool
}{
	{"0", 1e0, 0, false},
	{"0.", 1e0, 0, true},
	{"0.0", 1e0, 0, false},
	{"0.00", 1e0, 0, false},
	{"00.0", 1e0, 0, true},
	{"+0", 1e0, 0, true},
	{"1e0", 1e0, 0, true},
	{"1.000000000x", 1e9, 0, true},
	{"1.000000x", 1e6, 0, true},
	{"1.000x", 1e3, 0, true},
	{"1.x", 1e0, 0, true},
	{"1.0000000009", 1e9, +time.Second, false},
	{"1.0000009", 1e6, +time.Millisecond, false},
	{"1.0009", 1e3, +time.Microsecond, false},
	{"1.9", 1e0, +time.Nanosecond, false},
	{"-9223372036854775809", 1e0, 0, true},
	{"9223372036854775.808", 1e3, 0, true},
	{"-9223372036854.775809", 1e6, 0, true},
	{"9223372036.854775808", 1e9, 0, true},
	{"-1.9", 1e0, -time.Nanosecond, false},
	{"-1.0009", 1e3, -time.Microsecond, false},
	{"-1.0000009", 1e6, -time.Millisecond, false},
	{"-1.0000000009", 1e9, -time.Second, false},
}

func TestParseDuration(t *testing.T) {
	for _, tt := range parseDurationTestdata {
		a := durationArshaler{base: tt.base}
		switch err := a.unmarshal([]byte(tt.in)); {
		case a.td != tt.want:
			t.Errorf("parseDuration(%q, %s) = %v, want %v", tt.in, baseLabel(tt.base), a.td, tt.want)
		case (err == nil) && tt.wantErr:
			t.Errorf("parseDuration(%q, %s) error is nil, want non-nil", tt.in, baseLabel(tt.base))
		case (err != nil) && !tt.wantErr:
			t.Errorf("parseDuration(%q, %s) error is non-nil, want nil", tt.in, baseLabel(tt.base))
		}
	}
}

func FuzzFormatDuration(f *testing.F) {
	for _, tt := range formatDurationTestdata {
		f.Add(int64(tt.td))
	}
	f.Fuzz(func(t *testing.T, want int64) {
		var buf []byte
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9} {
			a := durationArshaler{td: time.Duration(want), base: base}
			buf, _ = a.appendMarshal(buf[:0])
			switch err := a.unmarshal(buf); {
			case err != nil:
				t.Fatalf("parseDuration(%q, %s) error: %v", buf, baseLabel(base), err)
			case a.td != time.Duration(want):
				t.Fatalf("parseDuration(%q, %s) = %v, want %v", buf, baseLabel(base), a.td, time.Duration(want))
			}
		}
	})
}

func FuzzParseDuration(f *testing.F) {
	for _, tt := range parseDurationTestdata {
		f.Add([]byte(tt.in))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9, 60} {
			a := durationArshaler{base: base}
			if err := a.unmarshal(in); err == nil && base != 60 {
				if n, err := jsonwire.ConsumeNumber(in); err != nil || n != len(in) {
					t.Fatalf("parseDuration(%q) error is nil for invalid JSON number", in)
				}
			}
		}
	})
}

type formatTimeTestdataEntry struct {
	ts        time.Time
	unixSec   string
	unixMilli string
	unixMicro string
	unixNano  string
}

var formatTimeTestdata = func() []formatTimeTestdataEntry {
	out := []formatTimeTestdataEntry{
		{time.Unix(math.MaxInt64/int64(1e0), 1e9-1).UTC(), "9223372036854775807.999999999", "9223372036854775807999.999999", "9223372036854775807999999.999", "9223372036854775807999999999"},
		{time.Unix(math.MaxInt64/int64(1e1), 1e9-1).UTC(), "922337203685477580.999999999", "922337203685477580999.999999", "922337203685477580999999.999", "922337203685477580999999999"},
		{time.Unix(math.MaxInt64/int64(1e2), 1e9-1).UTC(), "92233720368547758.999999999", "92233720368547758999.999999", "92233720368547758999999.999", "92233720368547758999999999"},
		{time.Unix(math.MinInt64, 1).UTC(), "-9223372036854775807.999999999", "-9223372036854775807999.999999", "-9223372036854775807999999.999", "-9223372036854775807999999999"},
		{time.Unix(math.MinInt64, 0).UTC(), "-9223372036854775808", "-9223372036854775808000", "-9223372036854775808000000", "-9223372036854775808000000000"},
	}
	for _, tt := range formatDurationTestdata {
		out = append(out, formatTimeTestdataEntry{time.Unix(0, int64(tt.td)).UTC(), tt.base10Sec, tt.base10Milli, tt.base10Micro, tt.base10Nano})
	}
	return out
}()

func TestFormatTime(t *testing.T) {
	var gotBuf []byte
	check := func(ts time.Time, s string, pow10 uint64) {
		gotBuf = appendTimeUnix(gotBuf[:0], ts, pow10)
		if string(gotBuf) != s {
			t.Errorf("formatTime(time.Unix(%d, %d), %s) = %q, want %q", ts.Unix(), ts.Nanosecond(), baseLabel(pow10), string(gotBuf), s)
		}
		gotTS, err := parseTimeUnix(gotBuf, pow10)
		if err != nil {
			t.Errorf("parseTime(%q, %s) error: %v", gotBuf, baseLabel(pow10), err)
		}
		if !gotTS.Equal(ts) {
			t.Errorf("parseTime(%q, %s) = time.Unix(%d, %d), want time.Unix(%d, %d)", gotBuf, baseLabel(pow10), gotTS.Unix(), gotTS.Nanosecond(), ts.Unix(), ts.Nanosecond())
		}
	}
	for _, tt := range formatTimeTestdata {
		check(tt.ts, tt.unixSec, 1e0)
		check(tt.ts, tt.unixMilli, 1e3)
		check(tt.ts, tt.unixMicro, 1e6)
		check(tt.ts, tt.unixNano, 1e9)
	}
}

var parseTimeTestdata = []struct {
	in      string
	base    uint64
	want    time.Time
	wantErr bool
}{
	{"0", 1e0, time.Unix(0, 0).UTC(), false},
	{"0.", 1e0, time.Time{}, true},
	{"0.0", 1e0, time.Unix(0, 0).UTC(), false},
	{"0.00", 1e0, time.Unix(0, 0).UTC(), false},
	{"00.0", 1e0, time.Time{}, true},
	{"+0", 1e0, time.Time{}, true},
	{"1e0", 1e0, time.Time{}, true},
	{"1234567890123456789012345678901234567890", 1e0, time.Time{}, true},
	{"9223372036854775808000.000000", 1e3, time.Time{}, true},
	{"9223372036854775807999999.9999", 1e6, time.Unix(math.MaxInt64, 1e9-1).UTC(), false},
	{"9223372036854775807999999999.9", 1e9, time.Unix(math.MaxInt64, 1e9-1).UTC(), false},
	{"9223372036854775807.999999999x", 1e0, time.Time{}, true},
	{"9223372036854775807000000000", 1e9, time.Unix(math.MaxInt64, 0).UTC(), false},
	{"-9223372036854775808", 1e0, time.Unix(math.MinInt64, 0).UTC(), false},
	{"-9223372036854775808000.000001", 1e3, time.Time{}, true},
	{"-9223372036854775808000000.0001", 1e6, time.Unix(math.MinInt64, 0).UTC(), false},
	{"-9223372036854775808000000000.x", 1e9, time.Time{}, true},
	{"-1234567890123456789012345678901234567890", 1e9, time.Time{}, true},
}

func TestParseTime(t *testing.T) {
	for _, tt := range parseTimeTestdata {
		a := timeArshaler{base: tt.base}
		switch err := a.unmarshal([]byte(tt.in)); {
		case a.tt != tt.want:
			t.Errorf("parseTime(%q, %s) = time.Unix(%d, %d), want time.Unix(%d, %d)", tt.in, baseLabel(tt.base), a.tt.Unix(), a.tt.Nanosecond(), tt.want.Unix(), tt.want.Nanosecond())
		case (err == nil) && tt.wantErr:
			t.Errorf("parseTime(%q, %s) = (time.Unix(%d, %d), nil), want non-nil error", tt.in, baseLabel(tt.base), a.tt.Unix(), a.tt.Nanosecond())
		case (err != nil) && !tt.wantErr:
			t.Errorf("parseTime(%q, %s) error is non-nil, want nil", tt.in, baseLabel(tt.base))
		}
	}
}

func FuzzFormatTime(f *testing.F) {
	for _, tt := range formatTimeTestdata {
		f.Add(tt.ts.Unix(), int64(tt.ts.Nanosecond()))
	}
	f.Fuzz(func(t *testing.T, wantSec, wantNano int64) {
		want := time.Unix(wantSec, int64(uint64(wantNano)%1e9)).UTC()
		var buf []byte
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9} {
			a := timeArshaler{tt: want, base: base}
			buf, _ = a.appendMarshal(buf[:0])
			switch err := a.unmarshal(buf); {
			case err != nil:
				t.Fatalf("parseTime(%q, %s) error: %v", buf, baseLabel(base), err)
			case a.tt != want:
				t.Fatalf("parseTime(%q, %s) = time.Unix(%d, %d), want time.Unix(%d, %d)", buf, baseLabel(base), a.tt.Unix(), a.tt.Nanosecond(), want.Unix(), want.Nanosecond())
			}
		}
	})
}

func FuzzParseTime(f *testing.F) {
	for _, tt := range parseTimeTestdata {
		f.Add([]byte(tt.in))
	}
	f.Fuzz(func(t *testing.T, in []byte) {
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9} {
			a := timeArshaler{base: base}
			if err := a.unmarshal(in); err == nil {
				if n, err := jsonwire.ConsumeNumber(in); err != nil || n != len(in) {
					t.Fatalf("parseTime(%q) error is nil for invalid JSON number", in)
				}
			}
		}
	})
}
