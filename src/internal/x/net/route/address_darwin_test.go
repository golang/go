// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package route

import (
	"reflect"
	"testing"
)

type parseAddrsOnDarwinTest struct {
	attrs uint
	fn    func(int, []byte) (int, Addr, error)
	b     []byte
	as    []Addr
}

var parseAddrsOnDarwinLittleEndianTests = []parseAddrsOnDarwinTest{
	{
		sysRTA_DST | sysRTA_GATEWAY | sysRTA_NETMASK,
		parseKernelInetAddr,
		[]byte{
			0x10, 0x2, 0x0, 0x0, 0xc0, 0xa8, 0x56, 0x0,
			0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,

			0x14, 0x12, 0x4, 0x0, 0x6, 0x0, 0x0, 0x0,
			0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
			0x0, 0x0, 0x0, 0x0,

			0x7, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		},
		[]Addr{
			&Inet4Addr{IP: [4]byte{192, 168, 86, 0}},
			&LinkAddr{Index: 4},
			&Inet4Addr{IP: [4]byte{255, 255, 255, 255}},
			nil,
			nil,
			nil,
			nil,
			nil,
		},
	},
}

func TestParseAddrsOnDarwin(t *testing.T) {
	tests := parseAddrsOnDarwinLittleEndianTests
	if nativeEndian != littleEndian {
		t.Skip("no test for non-little endian machine yet")
	}

	for i, tt := range tests {
		as, err := parseAddrs(tt.attrs, tt.fn, tt.b)
		if err != nil {
			t.Error(i, err)
			continue
		}
		if !reflect.DeepEqual(as, tt.as) {
			t.Errorf("#%d: got %+v; want %+v", i, as, tt.as)
			continue
		}
	}
}
