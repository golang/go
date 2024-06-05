// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	. "math/rand/v2"
	"testing"
)

func BenchmarkPCG_DXSM(b *testing.B) {
	var p PCG
	var t uint64
	for n := b.N; n > 0; n-- {
		t += p.Uint64()
	}
	Sink = t
}

func TestPCGMarshal(t *testing.T) {
	var p PCG
	const (
		seed1 = 0x123456789abcdef0
		seed2 = 0xfedcba9876543210
		want  = "pcg:\x12\x34\x56\x78\x9a\xbc\xde\xf0\xfe\xdc\xba\x98\x76\x54\x32\x10"
	)
	p.Seed(seed1, seed2)
	data, err := p.MarshalBinary()
	if string(data) != want || err != nil {
		t.Errorf("MarshalBinary() = %q, %v, want %q, nil", data, err, want)
	}

	q := PCG{}
	if err := q.UnmarshalBinary([]byte(want)); err != nil {
		t.Fatalf("UnmarshalBinary(): %v", err)
	}
	if q != p {
		t.Fatalf("after round trip, q = %#x, but p = %#x", q, p)
	}

	qu := q.Uint64()
	pu := p.Uint64()
	if qu != pu {
		t.Errorf("after round trip, q.Uint64() = %#x, but p.Uint64() = %#x", qu, pu)
	}
}

func TestPCG(t *testing.T) {
	p := NewPCG(1, 2)
	want := []uint64{
		0xc4f5a58656eef510,
		0x9dcec3ad077dec6c,
		0xc8d04605312f8088,
		0xcbedc0dcb63ac19a,
		0x3bf98798cae97950,
		0xa8c6d7f8d485abc,
		0x7ffa3780429cd279,
		0x730ad2626b1c2f8e,
		0x21ff2330f4a0ad99,
		0x2f0901a1947094b0,
		0xa9735a3cfbe36cef,
		0x71ddb0a01a12c84a,
		0xf0e53e77a78453bb,
		0x1f173e9663be1e9d,
		0x657651da3ac4115e,
		0xc8987376b65a157b,
		0xbb17008f5fca28e7,
		0x8232bd645f29ed22,
		0x12be8f07ad14c539,
		0x54908a48e8e4736e,
	}

	for i, x := range want {
		if u := p.Uint64(); u != x {
			t.Errorf("PCG #%d = %#x, want %#x", i, u, x)
		}
	}
}
