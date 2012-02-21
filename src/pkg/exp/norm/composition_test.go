// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "testing"

// TestCase is used for most tests.
type TestCase struct {
	in  []rune
	out []rune
}

type insertFunc func(rb *reorderBuffer, r rune) bool

func insert(rb *reorderBuffer, r rune) bool {
	src := inputString(string(r))
	return rb.insert(src, 0, rb.f.info(src, 0))
}

func runTests(t *testing.T, name string, fm Form, f insertFunc, tests []TestCase) {
	rb := reorderBuffer{}
	rb.init(fm, nil)
	for i, test := range tests {
		rb.reset()
		for j, rune := range test.in {
			b := []byte(string(rune))
			src := inputBytes(b)
			if !rb.insert(src, 0, rb.f.info(src, 0)) {
				t.Errorf("%s:%d: insert failed for rune %d", name, i, j)
			}
		}
		if rb.f.composing {
			rb.compose()
		}
		if rb.nrune != len(test.out) {
			t.Errorf("%s:%d: length = %d; want %d", name, i, rb.nrune, len(test.out))
			continue
		}
		for j, want := range test.out {
			found := rune(rb.runeAt(j))
			if found != want {
				t.Errorf("%s:%d: runeAt(%d) = %U; want %U", name, i, j, found, want)
			}
		}
	}
}

type flushFunc func(rb *reorderBuffer) []byte

func testFlush(t *testing.T, name string, fn flushFunc) {
	rb := reorderBuffer{}
	rb.init(NFC, nil)
	out := fn(&rb)
	if len(out) != 0 {
		t.Errorf("%s: wrote bytes on flush of empty buffer. (len(out) = %d)", name, len(out))
	}

	for _, r := range []rune("world!") {
		insert(&rb, r)
	}

	out = []byte("Hello ")
	out = rb.flush(out)
	want := "Hello world!"
	if string(out) != want {
		t.Errorf(`%s: output after flush was "%s"; want "%s"`, name, string(out), want)
	}
	if rb.nrune != 0 {
		t.Errorf("%s: non-null size of info buffer (rb.nrune == %d)", name, rb.nrune)
	}
	if rb.nbyte != 0 {
		t.Errorf("%s: non-null size of byte buffer (rb.nbyte == %d)", name, rb.nbyte)
	}
}

func flushF(rb *reorderBuffer) []byte {
	out := make([]byte, 0)
	return rb.flush(out)
}

func flushCopyF(rb *reorderBuffer) []byte {
	out := make([]byte, MaxSegmentSize)
	n := rb.flushCopy(out)
	return out[:n]
}

func TestFlush(t *testing.T) {
	testFlush(t, "flush", flushF)
	testFlush(t, "flushCopy", flushCopyF)
}

var insertTests = []TestCase{
	{[]rune{'a'}, []rune{'a'}},
	{[]rune{0x300}, []rune{0x300}},
	{[]rune{0x300, 0x316}, []rune{0x316, 0x300}}, // CCC(0x300)==230; CCC(0x316)==220
	{[]rune{0x316, 0x300}, []rune{0x316, 0x300}},
	{[]rune{0x41, 0x316, 0x300}, []rune{0x41, 0x316, 0x300}},
	{[]rune{0x41, 0x300, 0x316}, []rune{0x41, 0x316, 0x300}},
	{[]rune{0x300, 0x316, 0x41}, []rune{0x316, 0x300, 0x41}},
	{[]rune{0x41, 0x300, 0x40, 0x316}, []rune{0x41, 0x300, 0x40, 0x316}},
}

func TestInsert(t *testing.T) {
	runTests(t, "TestInsert", NFD, insert, insertTests)
}

var decompositionNFDTest = []TestCase{
	{[]rune{0xC0}, []rune{0x41, 0x300}},
	{[]rune{0xAC00}, []rune{0x1100, 0x1161}},
	{[]rune{0x01C4}, []rune{0x01C4}},
	{[]rune{0x320E}, []rune{0x320E}},
	{[]rune("음ẻ과"), []rune{0x110B, 0x1173, 0x11B7, 0x65, 0x309, 0x1100, 0x116A}},
}

var decompositionNFKDTest = []TestCase{
	{[]rune{0xC0}, []rune{0x41, 0x300}},
	{[]rune{0xAC00}, []rune{0x1100, 0x1161}},
	{[]rune{0x01C4}, []rune{0x44, 0x5A, 0x030C}},
	{[]rune{0x320E}, []rune{0x28, 0x1100, 0x1161, 0x29}},
}

func TestDecomposition(t *testing.T) {
	runTests(t, "TestDecompositionNFD", NFD, insert, decompositionNFDTest)
	runTests(t, "TestDecompositionNFKD", NFKD, insert, decompositionNFKDTest)
}

var compositionTest = []TestCase{
	{[]rune{0x41, 0x300}, []rune{0xC0}},
	{[]rune{0x41, 0x316}, []rune{0x41, 0x316}},
	{[]rune{0x41, 0x300, 0x35D}, []rune{0xC0, 0x35D}},
	{[]rune{0x41, 0x316, 0x300}, []rune{0xC0, 0x316}},
	// blocking starter
	{[]rune{0x41, 0x316, 0x40, 0x300}, []rune{0x41, 0x316, 0x40, 0x300}},
	{[]rune{0x1100, 0x1161}, []rune{0xAC00}},
	// parenthesized Hangul, alternate between ASCII and Hangul.
	{[]rune{0x28, 0x1100, 0x1161, 0x29}, []rune{0x28, 0xAC00, 0x29}},
}

func TestComposition(t *testing.T) {
	runTests(t, "TestComposition", NFC, insert, compositionTest)
}
