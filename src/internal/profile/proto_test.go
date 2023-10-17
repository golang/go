// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package profile

import (
	"reflect"
	"testing"
)

func TestPackedEncoding(t *testing.T) {

	type testcase struct {
		uint64s []uint64
		int64s  []int64
		encoded []byte
	}
	for i, tc := range []testcase{
		{
			[]uint64{0, 1, 10, 100, 1000, 10000},
			[]int64{1000, 0, 1000},
			[]byte{10, 8, 0, 1, 10, 100, 232, 7, 144, 78, 18, 5, 232, 7, 0, 232, 7},
		},
		{
			[]uint64{10000},
			nil,
			[]byte{8, 144, 78},
		},
		{
			nil,
			[]int64{-10000},
			[]byte{16, 240, 177, 255, 255, 255, 255, 255, 255, 255, 1},
		},
	} {
		source := &packedInts{tc.uint64s, tc.int64s}
		if got, want := marshal(source), tc.encoded; !reflect.DeepEqual(got, want) {
			t.Errorf("failed encode %d, got %v, want %v", i, got, want)
		}

		dest := new(packedInts)
		if err := unmarshal(tc.encoded, dest); err != nil {
			t.Errorf("failed decode %d: %v", i, err)
			continue
		}
		if got, want := dest.uint64s, tc.uint64s; !reflect.DeepEqual(got, want) {
			t.Errorf("failed decode uint64s %d, got %v, want %v", i, got, want)
		}
		if got, want := dest.int64s, tc.int64s; !reflect.DeepEqual(got, want) {
			t.Errorf("failed decode int64s %d, got %v, want %v", i, got, want)
		}
	}
}

type packedInts struct {
	uint64s []uint64
	int64s  []int64
}

func (u *packedInts) decoder() []decoder {
	return []decoder{
		nil,
		func(b *buffer, m message) error { return decodeUint64s(b, &m.(*packedInts).uint64s) },
		func(b *buffer, m message) error { return decodeInt64s(b, &m.(*packedInts).int64s) },
	}
}

func (u *packedInts) encode(b *buffer) {
	encodeUint64s(b, 1, u.uint64s)
	encodeInt64s(b, 2, u.int64s)
}
