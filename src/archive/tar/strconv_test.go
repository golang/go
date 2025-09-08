// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"math"
	"strings"
	"testing"
	"time"
)

func TestFitsInBase256(t *testing.T) {
	vectors := []struct {
		in    int64
		width int
		ok    bool
	}{
		{+1, 8, true},
		{0, 8, true},
		{-1, 8, true},
		{1 << 56, 8, false},
		{(1 << 56) - 1, 8, true},
		{-1 << 56, 8, true},
		{(-1 << 56) - 1, 8, false},
		{121654, 8, true},
		{-9849849, 8, true},
		{math.MaxInt64, 9, true},
		{0, 9, true},
		{math.MinInt64, 9, true},
		{math.MaxInt64, 12, true},
		{0, 12, true},
		{math.MinInt64, 12, true},
	}

	for _, v := range vectors {
		ok := fitsInBase256(v.width, v.in)
		if ok != v.ok {
			t.Errorf("fitsInBase256(%d, %d): got %v, want %v", v.in, v.width, ok, v.ok)
		}
	}
}

func TestParseNumeric(t *testing.T) {
	vectors := []struct {
		in   string
		want int64
		ok   bool
	}{
		// Test base-256 (binary) encoded values.
		{"", 0, true},
		{"\x80", 0, true},
		{"\x80\x00", 0, true},
		{"\x80\x00\x00", 0, true},
		{"\xbf", (1 << 6) - 1, true},
		{"\xbf\xff", (1 << 14) - 1, true},
		{"\xbf\xff\xff", (1 << 22) - 1, true},
		{"\xff", -1, true},
		{"\xff\xff", -1, true},
		{"\xff\xff\xff", -1, true},
		{"\xc0", -1 * (1 << 6), true},
		{"\xc0\x00", -1 * (1 << 14), true},
		{"\xc0\x00\x00", -1 * (1 << 22), true},
		{"\x87\x76\xa2\x22\xeb\x8a\x72\x61", 537795476381659745, true},
		{"\x80\x00\x00\x00\x07\x76\xa2\x22\xeb\x8a\x72\x61", 537795476381659745, true},
		{"\xf7\x76\xa2\x22\xeb\x8a\x72\x61", -615126028225187231, true},
		{"\xff\xff\xff\xff\xf7\x76\xa2\x22\xeb\x8a\x72\x61", -615126028225187231, true},
		{"\x80\x7f\xff\xff\xff\xff\xff\xff\xff", math.MaxInt64, true},
		{"\x80\x80\x00\x00\x00\x00\x00\x00\x00", 0, false},
		{"\xff\x80\x00\x00\x00\x00\x00\x00\x00", math.MinInt64, true},
		{"\xff\x7f\xff\xff\xff\xff\xff\xff\xff", 0, false},
		{"\xf5\xec\xd1\xc7\x7e\x5f\x26\x48\x81\x9f\x8f\x9b", 0, false},

		// Test base-8 (octal) encoded values.
		{"0000000\x00", 0, true},
		{" \x0000000\x00", 0, true},
		{" \x0000003\x00", 3, true},
		{"00000000227\x00", 0227, true},
		{"032033\x00 ", 032033, true},
		{"320330\x00 ", 0320330, true},
		{"0000660\x00 ", 0660, true},
		{"\x00 0000660\x00 ", 0660, true},
		{"0123456789abcdef", 0, false},
		{"0123456789\x00abcdef", 0, false},
		{"01234567\x0089abcdef", 342391, true},
		{"0123\x7e\x5f\x264123", 0, false},
	}

	for _, v := range vectors {
		var p parser
		got := p.parseNumeric([]byte(v.in))
		ok := (p.err == nil)
		if ok != v.ok {
			if v.ok {
				t.Errorf("parseNumeric(%q): got parsing failure, want success", v.in)
			} else {
				t.Errorf("parseNumeric(%q): got parsing success, want failure", v.in)
			}
		}
		if ok && got != v.want {
			t.Errorf("parseNumeric(%q): got %d, want %d", v.in, got, v.want)
		}
	}
}

func TestFormatNumeric(t *testing.T) {
	vectors := []struct {
		in   int64
		want string
		ok   bool
	}{
		// Test base-8 (octal) encoded values.
		{0, "0\x00", true},
		{7, "7\x00", true},
		{8, "\x80\x08", true},
		{077, "77\x00", true},
		{0100, "\x80\x00\x40", true},
		{0, "0000000\x00", true},
		{0123, "0000123\x00", true},
		{07654321, "7654321\x00", true},
		{07777777, "7777777\x00", true},
		{010000000, "\x80\x00\x00\x00\x00\x20\x00\x00", true},
		{0, "00000000000\x00", true},
		{000001234567, "00001234567\x00", true},
		{076543210321, "76543210321\x00", true},
		{012345670123, "12345670123\x00", true},
		{077777777777, "77777777777\x00", true},
		{0100000000000, "\x80\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00", true},
		{math.MaxInt64, "777777777777777777777\x00", true},

		// Test base-256 (binary) encoded values.
		{-1, "\xff", true},
		{-1, "\xff\xff", true},
		{-1, "\xff\xff\xff", true},
		{(1 << 0), "0", false},
		{(1 << 8) - 1, "\x80\xff", true},
		{(1 << 8), "0\x00", false},
		{(1 << 16) - 1, "\x80\xff\xff", true},
		{(1 << 16), "00\x00", false},
		{-1 * (1 << 0), "\xff", true},
		{-1*(1<<0) - 1, "0", false},
		{-1 * (1 << 8), "\xff\x00", true},
		{-1*(1<<8) - 1, "0\x00", false},
		{-1 * (1 << 16), "\xff\x00\x00", true},
		{-1*(1<<16) - 1, "00\x00", false},
		{537795476381659745, "0000000\x00", false},
		{537795476381659745, "\x80\x00\x00\x00\x07\x76\xa2\x22\xeb\x8a\x72\x61", true},
		{-615126028225187231, "0000000\x00", false},
		{-615126028225187231, "\xff\xff\xff\xff\xf7\x76\xa2\x22\xeb\x8a\x72\x61", true},
		{math.MaxInt64, "0000000\x00", false},
		{math.MaxInt64, "\x80\x00\x00\x00\x7f\xff\xff\xff\xff\xff\xff\xff", true},
		{math.MinInt64, "0000000\x00", false},
		{math.MinInt64, "\xff\xff\xff\xff\x80\x00\x00\x00\x00\x00\x00\x00", true},
		{math.MaxInt64, "\x80\x7f\xff\xff\xff\xff\xff\xff\xff", true},
		{math.MinInt64, "\xff\x80\x00\x00\x00\x00\x00\x00\x00", true},
	}

	for _, v := range vectors {
		var f formatter
		got := make([]byte, len(v.want))
		f.formatNumeric(got, v.in)
		ok := (f.err == nil)
		if ok != v.ok {
			if v.ok {
				t.Errorf("formatNumeric(%d): got formatting failure, want success", v.in)
			} else {
				t.Errorf("formatNumeric(%d): got formatting success, want failure", v.in)
			}
		}
		if string(got) != v.want {
			t.Errorf("formatNumeric(%d): got %q, want %q", v.in, got, v.want)
		}
	}
}

func TestFitsInOctal(t *testing.T) {
	vectors := []struct {
		input int64
		width int
		ok    bool
	}{
		{-1, 1, false},
		{-1, 2, false},
		{-1, 3, false},
		{0, 1, true},
		{0 + 1, 1, false},
		{0, 2, true},
		{07, 2, true},
		{07 + 1, 2, false},
		{0, 4, true},
		{0777, 4, true},
		{0777 + 1, 4, false},
		{0, 8, true},
		{07777777, 8, true},
		{07777777 + 1, 8, false},
		{0, 12, true},
		{077777777777, 12, true},
		{077777777777 + 1, 12, false},
		{math.MaxInt64, 22, true},
		{012345670123, 12, true},
		{01564164, 12, true},
		{-012345670123, 12, false},
		{-01564164, 12, false},
		{-1564164, 30, false},
	}

	for _, v := range vectors {
		ok := fitsInOctal(v.width, v.input)
		if ok != v.ok {
			t.Errorf("checkOctal(%d, %d): got %v, want %v", v.input, v.width, ok, v.ok)
		}
	}
}

func TestParsePAXTime(t *testing.T) {
	vectors := []struct {
		in   string
		want time.Time
		ok   bool
	}{
		{"1350244992.023960108", time.Unix(1350244992, 23960108), true},
		{"1350244992.02396010", time.Unix(1350244992, 23960100), true},
		{"1350244992.0239601089", time.Unix(1350244992, 23960108), true},
		{"1350244992.3", time.Unix(1350244992, 300000000), true},
		{"1350244992", time.Unix(1350244992, 0), true},
		{"-1.000000001", time.Unix(-1, -1e0+0e0), true},
		{"-1.000001", time.Unix(-1, -1e3+0e0), true},
		{"-1.001000", time.Unix(-1, -1e6+0e0), true},
		{"-1", time.Unix(-1, -0e0+0e0), true},
		{"-1.999000", time.Unix(-1, -1e9+1e6), true},
		{"-1.999999", time.Unix(-1, -1e9+1e3), true},
		{"-1.999999999", time.Unix(-1, -1e9+1e0), true},
		{"0.000000001", time.Unix(0, 1e0+0e0), true},
		{"0.000001", time.Unix(0, 1e3+0e0), true},
		{"0.001000", time.Unix(0, 1e6+0e0), true},
		{"0", time.Unix(0, 0e0), true},
		{"0.999000", time.Unix(0, 1e9-1e6), true},
		{"0.999999", time.Unix(0, 1e9-1e3), true},
		{"0.999999999", time.Unix(0, 1e9-1e0), true},
		{"1.000000001", time.Unix(+1, +1e0-0e0), true},
		{"1.000001", time.Unix(+1, +1e3-0e0), true},
		{"1.001000", time.Unix(+1, +1e6-0e0), true},
		{"1", time.Unix(+1, +0e0-0e0), true},
		{"1.999000", time.Unix(+1, +1e9-1e6), true},
		{"1.999999", time.Unix(+1, +1e9-1e3), true},
		{"1.999999999", time.Unix(+1, +1e9-1e0), true},
		{"-1350244992.023960108", time.Unix(-1350244992, -23960108), true},
		{"-1350244992.02396010", time.Unix(-1350244992, -23960100), true},
		{"-1350244992.0239601089", time.Unix(-1350244992, -23960108), true},
		{"-1350244992.3", time.Unix(-1350244992, -300000000), true},
		{"-1350244992", time.Unix(-1350244992, 0), true},
		{"", time.Time{}, false},
		{"0", time.Unix(0, 0), true},
		{"1.", time.Unix(1, 0), true},
		{"0.0", time.Unix(0, 0), true},
		{".5", time.Time{}, false},
		{"-1.3", time.Unix(-1, -3e8), true},
		{"-1.0", time.Unix(-1, -0e0), true},
		{"-0.0", time.Unix(-0, -0e0), true},
		{"-0.1", time.Unix(-0, -1e8), true},
		{"-0.01", time.Unix(-0, -1e7), true},
		{"-0.99", time.Unix(-0, -99e7), true},
		{"-0.98", time.Unix(-0, -98e7), true},
		{"-1.1", time.Unix(-1, -1e8), true},
		{"-1.01", time.Unix(-1, -1e7), true},
		{"-2.99", time.Unix(-2, -99e7), true},
		{"-5.98", time.Unix(-5, -98e7), true},
		{"-", time.Time{}, false},
		{"+", time.Time{}, false},
		{"-1.-1", time.Time{}, false},
		{"99999999999999999999999999999999999999999999999", time.Time{}, false},
		{"0.123456789abcdef", time.Time{}, false},
		{"foo", time.Time{}, false},
		{"\x00", time.Time{}, false},
		{"𝟵𝟴𝟳𝟲𝟱.𝟰𝟯𝟮𝟭𝟬", time.Time{}, false}, // Unicode numbers (U+1D7EC to U+1D7F5)
		{"98765﹒43210", time.Time{}, false}, // Unicode period (U+FE52)
	}

	for _, v := range vectors {
		ts, err := parsePAXTime(v.in)
		ok := (err == nil)
		if v.ok != ok {
			if v.ok {
				t.Errorf("parsePAXTime(%q): got parsing failure, want success", v.in)
			} else {
				t.Errorf("parsePAXTime(%q): got parsing success, want failure", v.in)
			}
		}
		if ok && !ts.Equal(v.want) {
			t.Errorf("parsePAXTime(%q): got (%ds %dns), want (%ds %dns)",
				v.in, ts.Unix(), ts.Nanosecond(), v.want.Unix(), v.want.Nanosecond())
		}
	}
}

func TestFormatPAXTime(t *testing.T) {
	vectors := []struct {
		sec, nsec int64
		want      string
	}{
		{1350244992, 0, "1350244992"},
		{1350244992, 300000000, "1350244992.3"},
		{1350244992, 23960100, "1350244992.0239601"},
		{1350244992, 23960108, "1350244992.023960108"},
		{+1, +1e9 - 1e0, "1.999999999"},
		{+1, +1e9 - 1e3, "1.999999"},
		{+1, +1e9 - 1e6, "1.999"},
		{+1, +0e0 - 0e0, "1"},
		{+1, +1e6 - 0e0, "1.001"},
		{+1, +1e3 - 0e0, "1.000001"},
		{+1, +1e0 - 0e0, "1.000000001"},
		{0, 1e9 - 1e0, "0.999999999"},
		{0, 1e9 - 1e3, "0.999999"},
		{0, 1e9 - 1e6, "0.999"},
		{0, 0e0, "0"},
		{0, 1e6 + 0e0, "0.001"},
		{0, 1e3 + 0e0, "0.000001"},
		{0, 1e0 + 0e0, "0.000000001"},
		{-1, -1e9 + 1e0, "-1.999999999"},
		{-1, -1e9 + 1e3, "-1.999999"},
		{-1, -1e9 + 1e6, "-1.999"},
		{-1, -0e0 + 0e0, "-1"},
		{-1, -1e6 + 0e0, "-1.001"},
		{-1, -1e3 + 0e0, "-1.000001"},
		{-1, -1e0 + 0e0, "-1.000000001"},
		{-1350244992, 0, "-1350244992"},
		{-1350244992, -300000000, "-1350244992.3"},
		{-1350244992, -23960100, "-1350244992.0239601"},
		{-1350244992, -23960108, "-1350244992.023960108"},
	}

	for _, v := range vectors {
		got := formatPAXTime(time.Unix(v.sec, v.nsec))
		if got != v.want {
			t.Errorf("formatPAXTime(%ds, %dns): got %q, want %q",
				v.sec, v.nsec, got, v.want)
		}
	}
}

func TestParsePAXRecord(t *testing.T) {
	medName := strings.Repeat("CD", 50)
	longName := strings.Repeat("AB", 100)

	vectors := []struct {
		in      string
		wantRes string
		wantKey string
		wantVal string
		ok      bool
	}{
		{"6 k=v\n\n", "\n", "k", "v", true},
		{"19 path=/etc/hosts\n", "", "path", "/etc/hosts", true},
		{"210 path=" + longName + "\nabc", "abc", "path", longName, true},
		{"110 path=" + medName + "\n", "", "path", medName, true},
		{"9 foo=ba\n", "", "foo", "ba", true},
		{"11 foo=bar\n\x00", "\x00", "foo", "bar", true},
		{"18 foo=b=\nar=\n==\x00\n", "", "foo", "b=\nar=\n==\x00", true},
		{"27 foo=hello9 foo=ba\nworld\n", "", "foo", "hello9 foo=ba\nworld", true},
		{"27 ☺☻☹=日a本b語ç\nmeow mix", "meow mix", "☺☻☹", "日a本b語ç", true},
		{"17 \x00hello=\x00world\n", "17 \x00hello=\x00world\n", "", "", false},
		{"1 k=1\n", "1 k=1\n", "", "", false},
		{"6 k~1\n", "6 k~1\n", "", "", false},
		{"6_k=1\n", "6_k=1\n", "", "", false},
		{"6 k=1 ", "6 k=1 ", "", "", false},
		{"632 k=1\n", "632 k=1\n", "", "", false},
		{"16 longkeyname=hahaha\n", "16 longkeyname=hahaha\n", "", "", false},
		{"3 somelongkey=\n", "3 somelongkey=\n", "", "", false},
		{"50 tooshort=\n", "50 tooshort=\n", "", "", false},
		{"0000000000000000000000000000000030 mtime=1432668921.098285006\n30 ctime=2147483649.15163319", "0000000000000000000000000000000030 mtime=1432668921.098285006\n30 ctime=2147483649.15163319", "mtime", "1432668921.098285006", false},
		{"06 k=v\n", "06 k=v\n", "", "", false},
		{"00006 k=v\n", "00006 k=v\n", "", "", false},
		{"000006 k=v\n", "000006 k=v\n", "", "", false},
		{"000000 k=v\n", "000000 k=v\n", "", "", false},
		{"0 k=v\n", "0 k=v\n", "", "", false},
		{"+0000005 x=\n", "+0000005 x=\n", "", "", false},
	}

	for _, v := range vectors {
		key, val, res, err := parsePAXRecord(v.in)
		ok := (err == nil)
		if ok != v.ok {
			if v.ok {
				t.Errorf("parsePAXRecord(%q): got parsing failure, want success", v.in)
			} else {
				t.Errorf("parsePAXRecord(%q): got parsing success, want failure", v.in)
			}
		}
		if v.ok && (key != v.wantKey || val != v.wantVal) {
			t.Errorf("parsePAXRecord(%q): got (%q: %q), want (%q: %q)",
				v.in, key, val, v.wantKey, v.wantVal)
		}
		if res != v.wantRes {
			t.Errorf("parsePAXRecord(%q): got residual %q, want residual %q",
				v.in, res, v.wantRes)
		}
	}
}

func TestFormatPAXRecord(t *testing.T) {
	medName := strings.Repeat("CD", 50)
	longName := strings.Repeat("AB", 100)

	vectors := []struct {
		inKey string
		inVal string
		want  string
		ok    bool
	}{
		{"k", "v", "6 k=v\n", true},
		{"path", "/etc/hosts", "19 path=/etc/hosts\n", true},
		{"path", longName, "210 path=" + longName + "\n", true},
		{"path", medName, "110 path=" + medName + "\n", true},
		{"foo", "ba", "9 foo=ba\n", true},
		{"foo", "bar", "11 foo=bar\n", true},
		{"foo", "b=\nar=\n==\x00", "18 foo=b=\nar=\n==\x00\n", true},
		{"foo", "hello9 foo=ba\nworld", "27 foo=hello9 foo=ba\nworld\n", true},
		{"☺☻☹", "日a本b語ç", "27 ☺☻☹=日a本b語ç\n", true},
		{"xhello", "\x00world", "17 xhello=\x00world\n", true},
		{"path", "null\x00", "", false},
		{"null\x00", "value", "", false},
		{paxSchilyXattr + "key", "null\x00", "26 SCHILY.xattr.key=null\x00\n", true},
	}

	for _, v := range vectors {
		got, err := formatPAXRecord(v.inKey, v.inVal)
		ok := (err == nil)
		if ok != v.ok {
			if v.ok {
				t.Errorf("formatPAXRecord(%q, %q): got format failure, want success", v.inKey, v.inVal)
			} else {
				t.Errorf("formatPAXRecord(%q, %q): got format success, want failure", v.inKey, v.inVal)
			}
		}
		if got != v.want {
			t.Errorf("formatPAXRecord(%q, %q): got %q, want %q",
				v.inKey, v.inVal, got, v.want)
		}
	}
}

func BenchmarkParsePAXTIme(b *testing.B) {
	tests := []struct {
		name string
		in   string
		want time.Time
		ok   bool
	}{
		{
			name: "NoNanos",
			in:   "123456",
			want: time.Unix(123456, 0),
			ok:   true,
		},
		{
			name: "ExactNanos",
			in:   "1.123456789",
			want: time.Unix(1, 123456789),
			ok:   true,
		},
		{
			name: "WithNanoPadding",
			in:   "1.123",
			want: time.Unix(1, 123000000),
			ok:   true,
		},
		{
			name: "WithNanoTruncate",
			in:   "1.123456789123",
			want: time.Unix(1, 123456789),
			ok:   true,
		},
		{
			name: "TrailingError",
			in:   "1.123abc",
			want: time.Time{},
			ok:   false,
		},
		{
			name: "LeadingError",
			in:   "1.abc123",
			want: time.Time{},
			ok:   false,
		},
	}
	for _, tt := range tests {
		b.Run(tt.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				ts, err := parsePAXTime(tt.in)
				if (err == nil) != tt.ok {
					if err != nil {
						b.Fatal(err)
					}
					b.Fatal("expected error")
				}
				if !ts.Equal(tt.want) {
					b.Fatalf("time mismatch: got %v, want %v", ts, tt.want)
				}
			}
		})
	}
}
