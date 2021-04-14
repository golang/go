// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bzip2

import (
	"bytes"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"testing"
)

func mustDecodeHex(s string) []byte {
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func mustLoadFile(f string) []byte {
	b, err := os.ReadFile(f)
	if err != nil {
		panic(err)
	}
	return b
}

func trim(b []byte) string {
	const limit = 1024
	if len(b) < limit {
		return fmt.Sprintf("%q", b)
	}
	return fmt.Sprintf("%q...", b[:limit])
}

func TestReader(t *testing.T) {
	var vectors = []struct {
		desc   string
		input  []byte
		output []byte
		fail   bool
	}{{
		desc: "hello world",
		input: mustDecodeHex("" +
			"425a68393141592653594eece83600000251800010400006449080200031064c" +
			"4101a7a9a580bb9431f8bb9229c28482776741b0",
		),
		output: []byte("hello world\n"),
	}, {
		desc: "concatenated files",
		input: mustDecodeHex("" +
			"425a68393141592653594eece83600000251800010400006449080200031064c" +
			"4101a7a9a580bb9431f8bb9229c28482776741b0425a68393141592653594eec" +
			"e83600000251800010400006449080200031064c4101a7a9a580bb9431f8bb92" +
			"29c28482776741b0",
		),
		output: []byte("hello world\nhello world\n"),
	}, {
		desc: "32B zeros",
		input: mustDecodeHex("" +
			"425a6839314159265359b5aa5098000000600040000004200021008283177245" +
			"385090b5aa5098",
		),
		output: make([]byte, 32),
	}, {
		desc: "1MiB zeros",
		input: mustDecodeHex("" +
			"425a683931415926535938571ce50008084000c0040008200030cc0529a60806" +
			"c4201e2ee48a70a12070ae39ca",
		),
		output: make([]byte, 1<<20),
	}, {
		desc:   "random data",
		input:  mustLoadFile("testdata/pass-random1.bz2"),
		output: mustLoadFile("testdata/pass-random1.bin"),
	}, {
		desc:   "random data - full symbol range",
		input:  mustLoadFile("testdata/pass-random2.bz2"),
		output: mustLoadFile("testdata/pass-random2.bin"),
	}, {
		desc: "random data - uses RLE1 stage",
		input: mustDecodeHex("" +
			"425a6839314159265359d992d0f60000137dfe84020310091c1e280e100e0428" +
			"01099210094806c0110002e70806402000546034000034000000f28300000320" +
			"00d3403264049270eb7a9280d308ca06ad28f6981bee1bf8160727c7364510d7" +
			"3a1e123083421b63f031f63993a0f40051fbf177245385090d992d0f60",
		),
		output: mustDecodeHex("" +
			"92d5652616ac444a4a04af1a8a3964aca0450d43d6cf233bd03233f4ba92f871" +
			"9e6c2a2bd4f5f88db07ecd0da3a33b263483db9b2c158786ad6363be35d17335" +
			"ba",
		),
	}, {
		desc:  "1MiB sawtooth",
		input: mustLoadFile("testdata/pass-sawtooth.bz2"),
		output: func() []byte {
			b := make([]byte, 1<<20)
			for i := range b {
				b[i] = byte(i)
			}
			return b
		}(),
	}, {
		desc:  "RLE2 buffer overrun - issue 5747",
		input: mustLoadFile("testdata/fail-issue5747.bz2"),
		fail:  true,
	}, {
		desc: "out-of-range selector - issue 8363",
		input: mustDecodeHex("" +
			"425a68393141592653594eece83600000251800010400006449080200031064c" +
			"4101a7a9a580bb943117724538509000000000",
		),
		fail: true,
	}, {
		desc: "bad block size - issue 13941",
		input: mustDecodeHex("" +
			"425a683131415926535936dc55330063ffc0006000200020a40830008b0008b8" +
			"bb9229c28481b6e2a998",
		),
		fail: true,
	}, {
		desc: "bad huffman delta",
		input: mustDecodeHex("" +
			"425a6836314159265359b1f7404b000000400040002000217d184682ee48a70a" +
			"12163ee80960",
		),
		fail: true,
	}}

	for i, v := range vectors {
		rd := NewReader(bytes.NewReader(v.input))
		buf, err := io.ReadAll(rd)

		if fail := bool(err != nil); fail != v.fail {
			if fail {
				t.Errorf("test %d (%s), unexpected failure: %v", i, v.desc, err)
			} else {
				t.Errorf("test %d (%s), unexpected success", i, v.desc)
			}
		}
		if !v.fail && !bytes.Equal(buf, v.output) {
			t.Errorf("test %d (%s), output mismatch:\ngot  %s\nwant %s", i, v.desc, trim(buf), trim(v.output))
		}
	}
}

func TestBitReader(t *testing.T) {
	var vectors = []struct {
		nbits uint // Number of bits to read
		value int  // Expected output value (0 for error)
		fail  bool // Expected operation failure?
	}{
		{nbits: 1, value: 1},
		{nbits: 1, value: 0},
		{nbits: 1, value: 1},
		{nbits: 5, value: 11},
		{nbits: 32, value: 0x12345678},
		{nbits: 15, value: 14495},
		{nbits: 3, value: 6},
		{nbits: 6, value: 13},
		{nbits: 1, fail: true},
	}

	rd := bytes.NewReader([]byte{0xab, 0x12, 0x34, 0x56, 0x78, 0x71, 0x3f, 0x8d})
	br := newBitReader(rd)
	for i, v := range vectors {
		val := br.ReadBits(v.nbits)
		if fail := bool(br.err != nil); fail != v.fail {
			if fail {
				t.Errorf("test %d, unexpected failure: ReadBits(%d) = %v", i, v.nbits, br.err)
			} else {
				t.Errorf("test %d, unexpected success: ReadBits(%d) = nil", i, v.nbits)
			}
		}
		if !v.fail && val != v.value {
			t.Errorf("test %d, mismatching value: ReadBits(%d) = %d, want %d", i, v.nbits, val, v.value)
		}
	}
}

func TestMTF(t *testing.T) {
	var vectors = []struct {
		idx int   // Input index
		sym uint8 // Expected output symbol
	}{
		{idx: 1, sym: 1}, // [1 0 2 3 4]
		{idx: 0, sym: 1}, // [1 0 2 3 4]
		{idx: 1, sym: 0}, // [0 1 2 3 4]
		{idx: 4, sym: 4}, // [4 0 1 2 3]
		{idx: 1, sym: 0}, // [0 4 1 2 3]
	}

	mtf := newMTFDecoderWithRange(5)
	for i, v := range vectors {
		sym := mtf.Decode(v.idx)
		t.Log(mtf)
		if sym != v.sym {
			t.Errorf("test %d, symbol mismatch: Decode(%d) = %d, want %d", i, v.idx, sym, v.sym)
		}
	}
}

func TestZeroRead(t *testing.T) {
	b := mustDecodeHex("425a6839314159265359b5aa5098000000600040000004200021008283177245385090b5aa5098")
	r := NewReader(bytes.NewReader(b))
	if n, err := r.Read(nil); n != 0 || err != nil {
		t.Errorf("Read(nil) = (%d, %v), want (0, nil)", n, err)
	}
}

var (
	digits = mustLoadFile("testdata/e.txt.bz2")
	newton = mustLoadFile("testdata/Isaac.Newton-Opticks.txt.bz2")
	random = mustLoadFile("testdata/random.data.bz2")
)

func benchmarkDecode(b *testing.B, compressed []byte) {
	// Determine the uncompressed size of testfile.
	uncompressedSize, err := io.Copy(io.Discard, NewReader(bytes.NewReader(compressed)))
	if err != nil {
		b.Fatal(err)
	}

	b.SetBytes(uncompressedSize)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(compressed)
		io.Copy(io.Discard, NewReader(r))
	}
}

func BenchmarkDecodeDigits(b *testing.B) { benchmarkDecode(b, digits) }
func BenchmarkDecodeNewton(b *testing.B) { benchmarkDecode(b, newton) }
func BenchmarkDecodeRand(b *testing.B)   { benchmarkDecode(b, random) }
