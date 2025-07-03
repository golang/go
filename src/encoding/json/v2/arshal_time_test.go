// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package json

import (
	"errors"
	"fmt"
	"math"
	"strconv"
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
	iso8601     string
}{
	{math.MaxInt64, "9223372036.854775807", "9223372036854.775807", "9223372036854775.807", "9223372036854775807", "PT2562047H47M16.854775807S"},
	{123*time.Hour + 4*time.Minute + 56*time.Second, "443096", "443096000", "443096000000", "443096000000000", "PT123H4M56S"},
	{time.Hour, "3600", "3600000", "3600000000", "3600000000000", "PT1H"},
	{time.Minute, "60", "60000", "60000000", "60000000000", "PT1M"},
	{1e12 + 1e12, "2000", "2000000", "2000000000", "2000000000000", "PT33M20S"},
	{1e12 + 1e11, "1100", "1100000", "1100000000", "1100000000000", "PT18M20S"},
	{1e12 + 1e10, "1010", "1010000", "1010000000", "1010000000000", "PT16M50S"},
	{1e12 + 1e9, "1001", "1001000", "1001000000", "1001000000000", "PT16M41S"},
	{1e12 + 1e8, "1000.1", "1000100", "1000100000", "1000100000000", "PT16M40.1S"},
	{1e12 + 1e7, "1000.01", "1000010", "1000010000", "1000010000000", "PT16M40.01S"},
	{1e12 + 1e6, "1000.001", "1000001", "1000001000", "1000001000000", "PT16M40.001S"},
	{1e12 + 1e5, "1000.0001", "1000000.1", "1000000100", "1000000100000", "PT16M40.0001S"},
	{1e12 + 1e4, "1000.00001", "1000000.01", "1000000010", "1000000010000", "PT16M40.00001S"},
	{1e12 + 1e3, "1000.000001", "1000000.001", "1000000001", "1000000001000", "PT16M40.000001S"},
	{1e12 + 1e2, "1000.0000001", "1000000.0001", "1000000000.1", "1000000000100", "PT16M40.0000001S"},
	{1e12 + 1e1, "1000.00000001", "1000000.00001", "1000000000.01", "1000000000010", "PT16M40.00000001S"},
	{1e12 + 1e0, "1000.000000001", "1000000.000001", "1000000000.001", "1000000000001", "PT16M40.000000001S"},
	{+(1e9 + 1), "1.000000001", "1000.000001", "1000000.001", "1000000001", "PT1.000000001S"},
	{+(1e9), "1", "1000", "1000000", "1000000000", "PT1S"},
	{+(1e9 - 1), "0.999999999", "999.999999", "999999.999", "999999999", "PT0.999999999S"},
	{+100000000, "0.1", "100", "100000", "100000000", "PT0.1S"},
	{+120000000, "0.12", "120", "120000", "120000000", "PT0.12S"},
	{+123000000, "0.123", "123", "123000", "123000000", "PT0.123S"},
	{+123400000, "0.1234", "123.4", "123400", "123400000", "PT0.1234S"},
	{+123450000, "0.12345", "123.45", "123450", "123450000", "PT0.12345S"},
	{+123456000, "0.123456", "123.456", "123456", "123456000", "PT0.123456S"},
	{+123456700, "0.1234567", "123.4567", "123456.7", "123456700", "PT0.1234567S"},
	{+123456780, "0.12345678", "123.45678", "123456.78", "123456780", "PT0.12345678S"},
	{+123456789, "0.123456789", "123.456789", "123456.789", "123456789", "PT0.123456789S"},
	{+12345678, "0.012345678", "12.345678", "12345.678", "12345678", "PT0.012345678S"},
	{+1234567, "0.001234567", "1.234567", "1234.567", "1234567", "PT0.001234567S"},
	{+123456, "0.000123456", "0.123456", "123.456", "123456", "PT0.000123456S"},
	{+12345, "0.000012345", "0.012345", "12.345", "12345", "PT0.000012345S"},
	{+1234, "0.000001234", "0.001234", "1.234", "1234", "PT0.000001234S"},
	{+123, "0.000000123", "0.000123", "0.123", "123", "PT0.000000123S"},
	{+12, "0.000000012", "0.000012", "0.012", "12", "PT0.000000012S"},
	{+1, "0.000000001", "0.000001", "0.001", "1", "PT0.000000001S"},
	{0, "0", "0", "0", "0", "PT0S"},
	{-1, "-0.000000001", "-0.000001", "-0.001", "-1", "-PT0.000000001S"},
	{-12, "-0.000000012", "-0.000012", "-0.012", "-12", "-PT0.000000012S"},
	{-123, "-0.000000123", "-0.000123", "-0.123", "-123", "-PT0.000000123S"},
	{-1234, "-0.000001234", "-0.001234", "-1.234", "-1234", "-PT0.000001234S"},
	{-12345, "-0.000012345", "-0.012345", "-12.345", "-12345", "-PT0.000012345S"},
	{-123456, "-0.000123456", "-0.123456", "-123.456", "-123456", "-PT0.000123456S"},
	{-1234567, "-0.001234567", "-1.234567", "-1234.567", "-1234567", "-PT0.001234567S"},
	{-12345678, "-0.012345678", "-12.345678", "-12345.678", "-12345678", "-PT0.012345678S"},
	{-123456789, "-0.123456789", "-123.456789", "-123456.789", "-123456789", "-PT0.123456789S"},
	{-123456780, "-0.12345678", "-123.45678", "-123456.78", "-123456780", "-PT0.12345678S"},
	{-123456700, "-0.1234567", "-123.4567", "-123456.7", "-123456700", "-PT0.1234567S"},
	{-123456000, "-0.123456", "-123.456", "-123456", "-123456000", "-PT0.123456S"},
	{-123450000, "-0.12345", "-123.45", "-123450", "-123450000", "-PT0.12345S"},
	{-123400000, "-0.1234", "-123.4", "-123400", "-123400000", "-PT0.1234S"},
	{-123000000, "-0.123", "-123", "-123000", "-123000000", "-PT0.123S"},
	{-120000000, "-0.12", "-120", "-120000", "-120000000", "-PT0.12S"},
	{-100000000, "-0.1", "-100", "-100000", "-100000000", "-PT0.1S"},
	{-(1e9 - 1), "-0.999999999", "-999.999999", "-999999.999", "-999999999", "-PT0.999999999S"},
	{-(1e9), "-1", "-1000", "-1000000", "-1000000000", "-PT1S"},
	{-(1e9 + 1), "-1.000000001", "-1000.000001", "-1000000.001", "-1000000001", "-PT1.000000001S"},
	{math.MinInt64, "-9223372036.854775808", "-9223372036854.775808", "-9223372036854775.808", "-9223372036854775808", "-PT2562047H47M16.854775808S"},
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
		check(tt.td, tt.iso8601, 8601)
	}
}

var parseDurationTestdata = []struct {
	in      string
	base    uint64
	want    time.Duration
	wantErr error
}{
	{"0", 1e0, 0, nil},
	{"0.", 1e0, 0, strconv.ErrSyntax},
	{"0.0", 1e0, 0, nil},
	{"0.00", 1e0, 0, nil},
	{"00.0", 1e0, 0, strconv.ErrSyntax},
	{"+0", 1e0, 0, strconv.ErrSyntax},
	{"1e0", 1e0, 0, strconv.ErrSyntax},
	{"1.000000000x", 1e9, 0, strconv.ErrSyntax},
	{"1.000000x", 1e6, 0, strconv.ErrSyntax},
	{"1.000x", 1e3, 0, strconv.ErrSyntax},
	{"1.x", 1e0, 0, strconv.ErrSyntax},
	{"1.0000000009", 1e9, +time.Second, nil},
	{"1.0000009", 1e6, +time.Millisecond, nil},
	{"1.0009", 1e3, +time.Microsecond, nil},
	{"1.9", 1e0, +time.Nanosecond, nil},
	{"-9223372036854775809", 1e0, 0, strconv.ErrRange},
	{"9223372036854775.808", 1e3, 0, strconv.ErrRange},
	{"-9223372036854.775809", 1e6, 0, strconv.ErrRange},
	{"9223372036.854775808", 1e9, 0, strconv.ErrRange},
	{"-1.9", 1e0, -time.Nanosecond, nil},
	{"-1.0009", 1e3, -time.Microsecond, nil},
	{"-1.0000009", 1e6, -time.Millisecond, nil},
	{"-1.0000000009", 1e9, -time.Second, nil},
	{"", 8601, 0, strconv.ErrSyntax},
	{"P", 8601, 0, strconv.ErrSyntax},
	{"PT", 8601, 0, strconv.ErrSyntax},
	{"PT0", 8601, 0, strconv.ErrSyntax},
	{"DT0S", 8601, 0, strconv.ErrSyntax},
	{"PT0S", 8601, 0, nil},
	{" PT0S", 8601, 0, strconv.ErrSyntax},
	{"PT0S ", 8601, 0, strconv.ErrSyntax},
	{"+PT0S", 8601, 0, nil},
	{"PT0.M", 8601, 0, strconv.ErrSyntax},
	{"PT0.S", 8601, 0, strconv.ErrSyntax},
	{"PT0.0S", 8601, 0, nil},
	{"PT0.0_0H", 8601, 0, strconv.ErrSyntax},
	{"PT0.0_0M", 8601, 0, strconv.ErrSyntax},
	{"PT0.0_0S", 8601, 0, strconv.ErrSyntax},
	{"PT.0S", 8601, 0, strconv.ErrSyntax},
	{"PT00.0S", 8601, 0, nil},
	{"PT0S", 8601, 0, nil},
	{"PT1,5S", 8601, time.Second + 500*time.Millisecond, nil},
	{"PT1H", 8601, time.Hour, nil},
	{"PT1H0S", 8601, time.Hour, nil},
	{"PT0S", 8601, 0, nil},
	{"PT00S", 8601, 0, nil},
	{"PT000S", 8601, 0, nil},
	{"PTS", 8601, 0, strconv.ErrSyntax},
	{"PT1M", 8601, time.Minute, nil},
	{"PT01M", 8601, time.Minute, nil},
	{"PT001M", 8601, time.Minute, nil},
	{"PT1H59S", 8601, time.Hour + 59*time.Second, nil},
	{"PT123H4M56.789S", 8601, 123*time.Hour + 4*time.Minute + 56*time.Second + 789*time.Millisecond, nil},
	{"-PT123H4M56.789S", 8601, -123*time.Hour - 4*time.Minute - 56*time.Second - 789*time.Millisecond, nil},
	{"PT0H0S", 8601, 0, nil},
	{"PT0H", 8601, 0, nil},
	{"PT0M", 8601, 0, nil},
	{"-PT0S", 8601, 0, nil},
	{"PT1M0S", 8601, time.Minute, nil},
	{"PT0H1M0S", 8601, time.Minute, nil},
	{"PT01H02M03S", 8601, 1*time.Hour + 2*time.Minute + 3*time.Second, nil},
	{"PT0,123S", 8601, 123 * time.Millisecond, nil},
	{"PT1.S", 8601, 0, strconv.ErrSyntax},
	{"PT1.000S", 8601, time.Second, nil},
	{"PT0.025H", 8601, time.Minute + 30*time.Second, nil},
	{"PT0.025H0M", 8601, 0, strconv.ErrSyntax},
	{"PT1.5M", 8601, time.Minute + 30*time.Second, nil},
	{"PT1.5M0S", 8601, 0, strconv.ErrSyntax},
	{"PT60M", 8601, time.Hour, nil},
	{"PT3600S", 8601, time.Hour, nil},
	{"PT1H2M3.0S", 8601, 1*time.Hour + 2*time.Minute + 3*time.Second, nil},
	{"pt1h2m3,0s", 8601, 1*time.Hour + 2*time.Minute + 3*time.Second, nil},
	{"PT-1H-2M-3S", 8601, 0, strconv.ErrSyntax},
	{"P1Y", 8601, time.Duration(daysPerYear * 24 * 60 * 60 * 1e9), errInaccurateDateUnits},
	{"P1.0Y", 8601, 0, strconv.ErrSyntax},
	{"P1M", 8601, time.Duration(daysPerYear / 12 * 24 * 60 * 60 * 1e9), errInaccurateDateUnits},
	{"P1.0M", 8601, 0, strconv.ErrSyntax},
	{"P1W", 8601, 7 * 24 * time.Hour, errInaccurateDateUnits},
	{"P1.0W", 8601, 0, strconv.ErrSyntax},
	{"P1D", 8601, 24 * time.Hour, errInaccurateDateUnits},
	{"P1.0D", 8601, 0, strconv.ErrSyntax},
	{"P1W1S", 8601, 0, strconv.ErrSyntax},
	{"-P1Y2M3W4DT5H6M7.8S", 8601, -(time.Duration(14*daysPerYear/12*24*60*60*1e9) + time.Duration((3*7+4)*24*60*60*1e9) + 5*time.Hour + 6*time.Minute + 7*time.Second + 800*time.Millisecond), errInaccurateDateUnits},
	{"-p1y2m3w4dt5h6m7.8s", 8601, -(time.Duration(14*daysPerYear/12*24*60*60*1e9) + time.Duration((3*7+4)*24*60*60*1e9) + 5*time.Hour + 6*time.Minute + 7*time.Second + 800*time.Millisecond), errInaccurateDateUnits},
	{"P0Y0M0DT1H2M3S", 8601, 1*time.Hour + 2*time.Minute + 3*time.Second, errInaccurateDateUnits},
	{"PT0.0000000001S", 8601, 0, nil},
	{"PT0.0000000005S", 8601, 0, nil},
	{"PT0.000000000500000000S", 8601, 0, nil},
	{"PT0.000000000499999999S", 8601, 0, nil},
	{"PT2562047H47M16.854775808S", 8601, 0, strconv.ErrRange},
	{"-PT2562047H47M16.854775809S", 8601, 0, strconv.ErrRange},
	{"PT9223372036.854775807S", 8601, math.MaxInt64, nil},
	{"PT9223372036.854775808S", 8601, 0, strconv.ErrRange},
	{"-PT9223372036.854775808S", 8601, math.MinInt64, nil},
	{"-PT9223372036.854775809S", 8601, 0, strconv.ErrRange},
	{"PT18446744073709551616S", 8601, 0, strconv.ErrRange},
	{"PT5124096H", 8601, 0, strconv.ErrRange},
	{"PT2562047.7880152155019444H", 8601, math.MaxInt64, nil},
	{"PT2562047.7880152155022222H", 8601, 0, strconv.ErrRange},
	{"PT5124094H94M33.709551616S", 8601, 0, strconv.ErrRange},
}

func TestParseDuration(t *testing.T) {
	for _, tt := range parseDurationTestdata {
		a := durationArshaler{base: tt.base}
		switch err := a.unmarshal([]byte(tt.in)); {
		case a.td != tt.want:
			t.Errorf("parseDuration(%q, %s) = %v, want %v", tt.in, baseLabel(tt.base), a.td, tt.want)
		case !errors.Is(err, tt.wantErr):
			t.Errorf("parseDuration(%q, %s) error = %v, want %v", tt.in, baseLabel(tt.base), err, tt.wantErr)
		}
	}
}

func FuzzFormatDuration(f *testing.F) {
	for _, tt := range formatDurationTestdata {
		f.Add(int64(tt.td))
	}
	f.Fuzz(func(t *testing.T, want int64) {
		var buf []byte
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9, 8601} {
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
		for _, base := range [...]uint64{1e0, 1e3, 1e6, 1e9, 8601} {
			a := durationArshaler{base: base}
			switch err := a.unmarshal(in); {
			case err != nil: // nothing else to check
			case base != 8601:
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
	wantErr error
}{
	{"0", 1e0, time.Unix(0, 0).UTC(), nil},
	{"0.", 1e0, time.Time{}, strconv.ErrSyntax},
	{"0.0", 1e0, time.Unix(0, 0).UTC(), nil},
	{"0.00", 1e0, time.Unix(0, 0).UTC(), nil},
	{"00.0", 1e0, time.Time{}, strconv.ErrSyntax},
	{"+0", 1e0, time.Time{}, strconv.ErrSyntax},
	{"1e0", 1e0, time.Time{}, strconv.ErrSyntax},
	{"1234567890123456789012345678901234567890", 1e0, time.Time{}, strconv.ErrRange},
	{"9223372036854775808000.000000", 1e3, time.Time{}, strconv.ErrRange},
	{"9223372036854775807999999.9999", 1e6, time.Unix(math.MaxInt64, 1e9-1).UTC(), nil},
	{"9223372036854775807999999999.9", 1e9, time.Unix(math.MaxInt64, 1e9-1).UTC(), nil},
	{"9223372036854775807.999999999x", 1e0, time.Time{}, strconv.ErrSyntax},
	{"9223372036854775807000000000", 1e9, time.Unix(math.MaxInt64, 0).UTC(), nil},
	{"-9223372036854775808", 1e0, time.Unix(math.MinInt64, 0).UTC(), nil},
	{"-9223372036854775808000.000001", 1e3, time.Time{}, strconv.ErrRange},
	{"-9223372036854775808000000.0001", 1e6, time.Unix(math.MinInt64, 0).UTC(), nil},
	{"-9223372036854775808000000000.x", 1e9, time.Time{}, strconv.ErrSyntax},
	{"-1234567890123456789012345678901234567890", 1e9, time.Time{}, strconv.ErrRange},
}

func TestParseTime(t *testing.T) {
	for _, tt := range parseTimeTestdata {
		a := timeArshaler{base: tt.base}
		switch err := a.unmarshal([]byte(tt.in)); {
		case a.tt != tt.want:
			t.Errorf("parseTime(%q, %s) = time.Unix(%d, %d), want time.Unix(%d, %d)", tt.in, baseLabel(tt.base), a.tt.Unix(), a.tt.Nanosecond(), tt.want.Unix(), tt.want.Nanosecond())
		case !errors.Is(err, tt.wantErr):
			t.Errorf("parseTime(%q, %s) error = %v, want %v", tt.in, baseLabel(tt.base), err, tt.wantErr)
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
