// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm

import "testing"

// TestCase is used for most tests.
type TestCase struct {
	in  []int
	out []int
}

type insertFunc func(rb *reorderBuffer, rune int) bool

func insert(rb *reorderBuffer, rune int) bool {
	src := inputString(string(rune))
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
			found := int(rb.runeAt(j))
			if found != want {
				t.Errorf("%s:%d: runeAt(%d) = %U; want %U", name, i, j, found, want)
			}
		}
	}
}

func TestFlush(t *testing.T) {
	rb := reorderBuffer{}
	rb.init(NFC, nil)
	out := make([]byte, 0)

	out = rb.flush(out)
	if len(out) != 0 {
		t.Errorf("wrote bytes on flush of empty buffer. (len(out) = %d)", len(out))
	}

	for _, r := range []int("world!") {
		insert(&rb, r)
	}

	out = []byte("Hello ")
	out = rb.flush(out)
	want := "Hello world!"
	if string(out) != want {
		t.Errorf(`output after flush was "%s"; want "%s"`, string(out), want)
	}
	if rb.nrune != 0 {
		t.Errorf("flush: non-null size of info buffer (rb.nrune == %d)", rb.nrune)
	}
	if rb.nbyte != 0 {
		t.Errorf("flush: non-null size of byte buffer (rb.nbyte == %d)", rb.nbyte)
	}
}

var insertTests = []TestCase{
	{[]int{'a'}, []int{'a'}},
	{[]int{0x300}, []int{0x300}},
	{[]int{0x300, 0x316}, []int{0x316, 0x300}}, // CCC(0x300)==230; CCC(0x316)==220
	{[]int{0x316, 0x300}, []int{0x316, 0x300}},
	{[]int{0x41, 0x316, 0x300}, []int{0x41, 0x316, 0x300}},
	{[]int{0x41, 0x300, 0x316}, []int{0x41, 0x316, 0x300}},
	{[]int{0x300, 0x316, 0x41}, []int{0x316, 0x300, 0x41}},
	{[]int{0x41, 0x300, 0x40, 0x316}, []int{0x41, 0x300, 0x40, 0x316}},
}

func TestInsert(t *testing.T) {
	runTests(t, "TestInsert", NFD, insert, insertTests)
}

var decompositionNFDTest = []TestCase{
	{[]int{0xC0}, []int{0x41, 0x300}},
	{[]int{0xAC00}, []int{0x1100, 0x1161}},
	{[]int{0x01C4}, []int{0x01C4}},
	{[]int{0x320E}, []int{0x320E}},
	{[]int("음ẻ과"), []int{0x110B, 0x1173, 0x11B7, 0x65, 0x309, 0x1100, 0x116A}},
}

var decompositionNFKDTest = []TestCase{
	{[]int{0xC0}, []int{0x41, 0x300}},
	{[]int{0xAC00}, []int{0x1100, 0x1161}},
	{[]int{0x01C4}, []int{0x44, 0x5A, 0x030C}},
	{[]int{0x320E}, []int{0x28, 0x1100, 0x1161, 0x29}},
}

func TestDecomposition(t *testing.T) {
	runTests(t, "TestDecompositionNFD", NFD, insert, decompositionNFDTest)
	runTests(t, "TestDecompositionNFKD", NFKD, insert, decompositionNFKDTest)
}

var compositionTest = []TestCase{
	{[]int{0x41, 0x300}, []int{0xC0}},
	{[]int{0x41, 0x316}, []int{0x41, 0x316}},
	{[]int{0x41, 0x300, 0x35D}, []int{0xC0, 0x35D}},
	{[]int{0x41, 0x316, 0x300}, []int{0xC0, 0x316}},
	// blocking starter
	{[]int{0x41, 0x316, 0x40, 0x300}, []int{0x41, 0x316, 0x40, 0x300}},
	{[]int{0x1100, 0x1161}, []int{0xAC00}},
	// parenthesized Hangul, alternate between ASCII and Hangul.
	{[]int{0x28, 0x1100, 0x1161, 0x29}, []int{0x28, 0xAC00, 0x29}},
}

func TestComposition(t *testing.T) {
	runTests(t, "TestComposition", NFC, insert, compositionTest)
}
