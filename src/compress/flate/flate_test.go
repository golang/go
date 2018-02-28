// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test tests some internals of the flate package.
// The tests in package compress/gzip serve as the
// end-to-end test of the decompressor.

package flate

import (
	"bytes"
	"encoding/hex"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

// The following test should not panic.
func TestIssue5915(t *testing.T) {
	bits := []int{4, 0, 0, 6, 4, 3, 2, 3, 3, 4, 4, 5, 0, 0, 0, 0, 5, 5, 6,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 6, 0, 11, 0, 8, 0, 6, 6, 10, 8}
	var h huffmanDecoder
	if h.init(bits) {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue5962(t *testing.T) {
	bits := []int{4, 0, 0, 6, 4, 3, 2, 3, 3, 4, 4, 5, 0, 0, 0, 0,
		5, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11}
	var h huffmanDecoder
	if h.init(bits) {
		t.Fatalf("Given sequence of bits is bad, and should not succeed.")
	}
}

// The following test should not panic.
func TestIssue6255(t *testing.T) {
	bits1 := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11}
	bits2 := []int{11, 13}
	var h huffmanDecoder
	if !h.init(bits1) {
		t.Fatalf("Given sequence of bits is good and should succeed.")
	}
	if h.init(bits2) {
		t.Fatalf("Given sequence of bits is bad and should not succeed.")
	}
}

func TestInvalidEncoding(t *testing.T) {
	// Initialize Huffman decoder to recognize "0".
	var h huffmanDecoder
	if !h.init([]int{1}) {
		t.Fatal("Failed to initialize Huffman decoder")
	}

	// Initialize decompressor with invalid Huffman coding.
	var f decompressor
	f.r = bytes.NewReader([]byte{0xff})

	_, err := f.huffSym(&h)
	if err == nil {
		t.Fatal("Should have rejected invalid bit sequence")
	}
}

func TestInvalidBits(t *testing.T) {
	oversubscribed := []int{1, 2, 3, 4, 4, 5}
	incomplete := []int{1, 2, 4, 4}
	var h huffmanDecoder
	if h.init(oversubscribed) {
		t.Fatal("Should reject oversubscribed bit-length set")
	}
	if h.init(incomplete) {
		t.Fatal("Should reject incomplete bit-length set")
	}
}

func TestStreams(t *testing.T) {
	// To verify any of these hexstrings as valid or invalid flate streams
	// according to the C zlib library, you can use the Python wrapper library:
	// >>> hex_string = "010100feff11"
	// >>> import zlib
	// >>> zlib.decompress(hex_string.decode("hex"), -15) # Negative means raw DEFLATE
	// '\x11'

	testCases := []struct {
		desc   string // Description of the stream
		stream string // Hexstring of the input DEFLATE stream
		want   string // Expected result. Use "fail" to expect failure
	}{{
		"degenerate HCLenTree",
		"05e0010000000000100000000000000000000000000000000000000000000000" +
			"00000000000000000004",
		"fail",
	}, {
		"complete HCLenTree, empty HLitTree, empty HDistTree",
		"05e0010400000000000000000000000000000000000000000000000000000000" +
			"00000000000000000010",
		"fail",
	}, {
		"empty HCLenTree",
		"05e0010000000000000000000000000000000000000000000000000000000000" +
			"00000000000000000010",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree, empty HDistTree, use missing HDist symbol",
		"000100feff000de0010400000000100000000000000000000000000000000000" +
			"0000000000000000000000000000002c",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree, degenerate HDistTree, use missing HDist symbol",
		"000100feff000de0010000000000000000000000000000000000000000000000" +
			"00000000000000000610000000004070",
		"fail",
	}, {
		"complete HCLenTree, empty HLitTree, empty HDistTree",
		"05e0010400000000100400000000000000000000000000000000000000000000" +
			"0000000000000000000000000008",
		"fail",
	}, {
		"complete HCLenTree, empty HLitTree, degenerate HDistTree",
		"05e0010400000000100400000000000000000000000000000000000000000000" +
			"0000000000000000000800000008",
		"fail",
	}, {
		"complete HCLenTree, degenerate HLitTree, degenerate HDistTree, use missing HLit symbol",
		"05e0010400000000100000000000000000000000000000000000000000000000" +
			"0000000000000000001c",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree, too large HDistTree",
		"edff870500000000200400000000000000000000000000000000000000000000" +
			"000000000000000000080000000000000004",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree, empty HDistTree, excessive repeater code",
		"edfd870500000000200400000000000000000000000000000000000000000000" +
			"000000000000000000e8b100",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree, empty HDistTree of normal length 30",
		"05fd01240000000000f8ffffffffffffffffffffffffffffffffffffffffffff" +
			"ffffffffffffffffff07000000fe01",
		"",
	}, {
		"complete HCLenTree, complete HLitTree, empty HDistTree of excessive length 31",
		"05fe01240000000000f8ffffffffffffffffffffffffffffffffffffffffffff" +
			"ffffffffffffffffff07000000fc03",
		"fail",
	}, {
		"complete HCLenTree, over-subscribed HLitTree, empty HDistTree",
		"05e001240000000000fcffffffffffffffffffffffffffffffffffffffffffff" +
			"ffffffffffffffffff07f00f",
		"fail",
	}, {
		"complete HCLenTree, under-subscribed HLitTree, empty HDistTree",
		"05e001240000000000fcffffffffffffffffffffffffffffffffffffffffffff" +
			"fffffffffcffffffff07f00f",
		"fail",
	}, {
		"complete HCLenTree, complete HLitTree with single code, empty HDistTree",
		"05e001240000000000f8ffffffffffffffffffffffffffffffffffffffffffff" +
			"ffffffffffffffffff07f00f",
		"01",
	}, {
		"complete HCLenTree, complete HLitTree with multiple codes, empty HDistTree",
		"05e301240000000000f8ffffffffffffffffffffffffffffffffffffffffffff" +
			"ffffffffffffffffff07807f",
		"01",
	}, {
		"complete HCLenTree, complete HLitTree, degenerate HDistTree, use valid HDist symbol",
		"000100feff000de0010400000000100000000000000000000000000000000000" +
			"0000000000000000000000000000003c",
		"00000000",
	}, {
		"complete HCLenTree, degenerate HLitTree, degenerate HDistTree",
		"05e0010400000000100000000000000000000000000000000000000000000000" +
			"0000000000000000000c",
		"",
	}, {
		"complete HCLenTree, degenerate HLitTree, empty HDistTree",
		"05e0010400000000100000000000000000000000000000000000000000000000" +
			"00000000000000000004",
		"",
	}, {
		"complete HCLenTree, complete HLitTree, empty HDistTree, spanning repeater code",
		"edfd870500000000200400000000000000000000000000000000000000000000" +
			"000000000000000000e8b000",
		"",
	}, {
		"complete HCLenTree with length codes, complete HLitTree, empty HDistTree",
		"ede0010400000000100000000000000000000000000000000000000000000000" +
			"0000000000000000000400004000",
		"",
	}, {
		"complete HCLenTree, complete HLitTree, degenerate HDistTree, use valid HLit symbol 284 with count 31",
		"000100feff00ede0010400000000100000000000000000000000000000000000" +
			"000000000000000000000000000000040000407f00",
		"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"0000000000000000000000000000000000000000000000000000000000000000" +
			"000000",
	}, {
		"complete HCLenTree, complete HLitTree, degenerate HDistTree, use valid HLit and HDist symbols",
		"0cc2010d00000082b0ac4aff0eb07d27060000ffff",
		"616263616263",
	}, {
		"fixed block, use reserved symbol 287",
		"33180700",
		"fail",
	}, {
		"raw block",
		"010100feff11",
		"11",
	}, {
		"issue 10426 - over-subscribed HCLenTree causes a hang",
		"344c4a4e494d4b070000ff2e2eff2e2e2e2e2eff",
		"fail",
	}, {
		"issue 11030 - empty HDistTree unexpectedly leads to error",
		"05c0070600000080400fff37a0ca",
		"",
	}, {
		"issue 11033 - empty HDistTree unexpectedly leads to error",
		"050fb109c020cca5d017dcbca044881ee1034ec149c8980bbc413c2ab35be9dc" +
			"b1473449922449922411202306ee97b0383a521b4ffdcf3217f9f7d3adb701",
		"3130303634342068652e706870005d05355f7ed957ff084a90925d19e3ebc6d0" +
			"c6d7",
	}}

	for i, tc := range testCases {
		data, err := hex.DecodeString(tc.stream)
		if err != nil {
			t.Fatal(err)
		}
		data, err = ioutil.ReadAll(NewReader(bytes.NewReader(data)))
		if tc.want == "fail" {
			if err == nil {
				t.Errorf("#%d (%s): got nil error, want non-nil", i, tc.desc)
			}
		} else {
			if err != nil {
				t.Errorf("#%d (%s): %v", i, tc.desc, err)
				continue
			}
			if got := hex.EncodeToString(data); got != tc.want {
				t.Errorf("#%d (%s):\ngot  %q\nwant %q", i, tc.desc, got, tc.want)
			}

		}
	}
}

func TestTruncatedStreams(t *testing.T) {
	const data = "\x00\f\x00\xf3\xffhello, world\x01\x00\x00\xff\xff"

	for i := 0; i < len(data)-1; i++ {
		r := NewReader(strings.NewReader(data[:i]))
		_, err := io.Copy(ioutil.Discard, r)
		if err != io.ErrUnexpectedEOF {
			t.Errorf("io.Copy(%d) on truncated stream: got %v, want %v", i, err, io.ErrUnexpectedEOF)
		}
	}
}

// Verify that flate.Reader.Read returns (n, io.EOF) instead
// of (n, nil) + (0, io.EOF) when possible.
//
// This helps net/http.Transport reuse HTTP/1 connections more
// aggressively.
//
// See https://github.com/google/go-github/pull/317 for background.
func TestReaderEarlyEOF(t *testing.T) {
	t.Parallel()
	testSizes := []int{
		1, 2, 3, 4, 5, 6, 7, 8,
		100, 1000, 10000, 100000,
		128, 1024, 16384, 131072,

		// Testing multiples of windowSize triggers the case
		// where Read will fail to return an early io.EOF.
		windowSize * 1, windowSize * 2, windowSize * 3,
	}

	var maxSize int
	for _, n := range testSizes {
		if maxSize < n {
			maxSize = n
		}
	}

	readBuf := make([]byte, 40)
	data := make([]byte, maxSize)
	for i := range data {
		data[i] = byte(i)
	}

	for _, sz := range testSizes {
		if testing.Short() && sz > windowSize {
			continue
		}
		for _, flush := range []bool{true, false} {
			earlyEOF := true // Do we expect early io.EOF?

			var buf bytes.Buffer
			w, _ := NewWriter(&buf, 5)
			w.Write(data[:sz])
			if flush {
				// If a Flush occurs after all the actual data, the flushing
				// semantics dictate that we will observe a (0, io.EOF) since
				// Read must return data before it knows that the stream ended.
				w.Flush()
				earlyEOF = false
			}
			w.Close()

			r := NewReader(&buf)
			for {
				n, err := r.Read(readBuf)
				if err == io.EOF {
					// If the availWrite == windowSize, then that means that the
					// previous Read returned because the write buffer was full
					// and it just so happened that the stream had no more data.
					// This situation is rare, but unavoidable.
					if r.(*decompressor).dict.availWrite() == windowSize {
						earlyEOF = false
					}

					if n == 0 && earlyEOF {
						t.Errorf("On size:%d flush:%v, Read() = (0, io.EOF), want (n, io.EOF)", sz, flush)
					}
					if n != 0 && !earlyEOF {
						t.Errorf("On size:%d flush:%v, Read() = (%d, io.EOF), want (0, io.EOF)", sz, flush, n)
					}
					break
				}
				if err != nil {
					t.Fatal(err)
				}
			}
		}
	}
}
