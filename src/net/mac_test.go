// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"strings"
	"testing"
)

var parseMACTests = []struct {
	in  string
	out HardwareAddr
	err string
}{
	// See RFC 7042, Section 2.1.1.
	{"00:00:5e:00:53:01", HardwareAddr{0x00, 0x00, 0x5e, 0x00, 0x53, 0x01}, ""},
	{"00-00-5e-00-53-01", HardwareAddr{0x00, 0x00, 0x5e, 0x00, 0x53, 0x01}, ""},
	{"0000.5e00.5301", HardwareAddr{0x00, 0x00, 0x5e, 0x00, 0x53, 0x01}, ""},
	{"00005e005301", HardwareAddr{0x00, 0x00, 0x5e, 0x00, 0x53, 0x01}, ""},

	// See RFC 7042, Section 2.2.2.
	{"02:00:5e:10:00:00:00:01", HardwareAddr{0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01}, ""},
	{"02-00-5e-10-00-00-00-01", HardwareAddr{0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01}, ""},
	{"0200.5e10.0000.0001", HardwareAddr{0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01}, ""},
	{"02005e1000000001", HardwareAddr{0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01}, ""},

	// See RFC 4391, Section 9.1.1.
	{
		"00:00:00:00:fe:80:00:00:00:00:00:00:02:00:5e:10:00:00:00:01",
		HardwareAddr{
			0x00, 0x00, 0x00, 0x00,
			0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01,
		},
		"",
	},
	{
		"00-00-00-00-fe-80-00-00-00-00-00-00-02-00-5e-10-00-00-00-01",
		HardwareAddr{
			0x00, 0x00, 0x00, 0x00,
			0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01,
		},
		"",
	},
	{
		"0000.0000.fe80.0000.0000.0000.0200.5e10.0000.0001",
		HardwareAddr{
			0x00, 0x00, 0x00, 0x00,
			0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01,
		},
		"",
	},
	{
		"00000000fe8000000000000002005e1000000001",
		HardwareAddr{
			0x00, 0x00, 0x00, 0x00,
			0xfe, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
			0x02, 0x00, 0x5e, 0x10, 0x00, 0x00, 0x00, 0x01,
		},
		"",
	},

	{"ab:cd:ef:AB:CD:EF", HardwareAddr{0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef}, ""},
	{"ab:cd:ef:AB:CD:EF:ab:cd", HardwareAddr{0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef, 0xab, 0xcd}, ""},
	{
		"ab:cd:ef:AB:CD:EF:ab:cd:ef:AB:CD:EF:ab:cd:ef:AB:CD:EF:ab:cd",
		HardwareAddr{
			0xab, 0xcd, 0xef, 0xab,
			0xcd, 0xef, 0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef,
			0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef, 0xab, 0xcd,
		},
		"",
	},

	{"01.02.03.04.05.06", nil, "invalid MAC address"},
	{"01:02:03:04:05:06:", nil, "invalid MAC address"},
	{"x1:02:03:04:05:06", nil, "invalid MAC address"},
	{"01002:03:04:05:06", nil, "invalid MAC address"},
	{"01:02003:04:05:06", nil, "invalid MAC address"},
	{"01:02:03004:05:06", nil, "invalid MAC address"},
	{"01:02:03:04005:06", nil, "invalid MAC address"},
	{"01:02:03:04:05006", nil, "invalid MAC address"},
	{"01-02:03:04:05:06", nil, "invalid MAC address"},
	{"01:02-03-04-05-06", nil, "invalid MAC address"},
	{"0123:4567:89AF", nil, "invalid MAC address"},
	{"0123-4567-89AF", nil, "invalid MAC address"},
	{"0123456789AF0", nil, "invalid MAC address"},
}

func TestParseMAC(t *testing.T) {
	match := func(err error, s string) bool {
		if s == "" {
			return err == nil
		}
		return err != nil && strings.Contains(err.Error(), s)
	}

	for i, tt := range parseMACTests {
		out, err := ParseMAC(tt.in)
		if !reflect.DeepEqual(out, tt.out) || !match(err, tt.err) {
			t.Errorf("ParseMAC(%q) = %v, %v, want %v, %v", tt.in, out, err, tt.out, tt.err)
		}
		if tt.err == "" {
			// Verify that serialization works too, and that it round-trips.
			s := out.String()
			out2, err := ParseMAC(s)
			if err != nil {
				t.Errorf("%d. ParseMAC(%q) = %v", i, s, err)
				continue
			}
			if !reflect.DeepEqual(out2, out) {
				t.Errorf("%d. ParseMAC(%q) = %v, want %v", i, s, out2, out)
			}
		}
	}
}
