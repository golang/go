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
	{"01:23:45:67:89:AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"01-23-45-67-89-AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"0123.4567.89AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"ab:cd:ef:AB:CD:EF", HardwareAddr{0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef}, ""},
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
	{"01:23:45:67:89:AB:CD:EF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
	{"01-23-45-67-89-AB-CD-EF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
	{"0123.4567.89AB.CDEF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
	{
		"01:23:45:67:89:ab:cd:ef:00:00:01:23:45:67:89:ab:cd:ef:00:00",
		HardwareAddr{
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
		},
		"",
	},
	{
		"01-23-45-67-89-ab-cd-ef-00-00-01-23-45-67-89-ab-cd-ef-00-00",
		HardwareAddr{
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
		},
		"",
	},
	{
		"0123.4567.89ab.cdef.0000.0123.4567.89ab.cdef.0000",
		HardwareAddr{
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
			0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0x00, 0x00,
		},
		"",
	},
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
