// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"strings"
	"testing"
)

var mactests = []struct {
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
}

func match(err error, s string) bool {
	if s == "" {
		return err == nil
	}
	return err != nil && strings.Contains(err.Error(), s)
}

func TestParseMAC(t *testing.T) {
	for _, tt := range mactests {
		out, err := ParseMAC(tt.in)
		if !reflect.DeepEqual(out, tt.out) || !match(err, tt.err) {
			t.Errorf("ParseMAC(%q) = %v, %v, want %v, %v", tt.in, out, err, tt.out,
				tt.err)
		}
	}
}
