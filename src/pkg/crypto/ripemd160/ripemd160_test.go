// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ripemd160

// Test vectors are from:
// http://homes.esat.kuleuven.be/~bosselae/ripemd160.html

import (
	"fmt"
	"io"
	"testing"
)

type mdTest struct {
	out string
	in  string
}

var vectors = [...]mdTest{
	{"9c1185a5c5e9fc54612808977ee8f548b2258d31", ""},
	{"0bdc9d2d256b3ee9daae347be6f4dc835a467ffe", "a"},
	{"8eb208f7e05d987a9b044a8e98c6b087f15a0bfc", "abc"},
	{"5d0689ef49d2fae572b881b123a85ffa21595f36", "message digest"},
	{"f71c27109c692c1b56bbdceb5b9d2865b3708dbc", "abcdefghijklmnopqrstuvwxyz"},
	{"12a053384a9c0c88e405a06c27dcf49ada62eb2b", "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"},
	{"b0e20b6e3116640286ed3a87a5713079b21f5189", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"},
	{"9b752e45573d4b39f4dbd3323cab82bf63326bfb", "12345678901234567890123456789012345678901234567890123456789012345678901234567890"},
}

func TestVectors(t *testing.T) {
	for i := 0; i < len(vectors); i++ {
		tv := vectors[i]
		md := New()
		for j := 0; j < 3; j++ {
			if j < 2 {
				io.WriteString(md, tv.in)
			} else {
				io.WriteString(md, tv.in[0:len(tv.in)/2])
				md.Sum(nil)
				io.WriteString(md, tv.in[len(tv.in)/2:])
			}
			s := fmt.Sprintf("%x", md.Sum(nil))
			if s != tv.out {
				t.Fatalf("RIPEMD-160[%d](%s) = %s, expected %s", j, tv.in, s, tv.out)
			}
			md.Reset()
		}
	}
}

func TestMillionA(t *testing.T) {
	md := New()
	for i := 0; i < 100000; i++ {
		io.WriteString(md, "aaaaaaaaaa")
	}
	out := "52783243c1697bdbe16d37f97f68f08325dc1528"
	s := fmt.Sprintf("%x", md.Sum(nil))
	if s != out {
		t.Fatalf("RIPEMD-160 (1 million 'a') = %s, expected %s", s, out)
	}
	md.Reset()
}
