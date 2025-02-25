// errorcheck -0 -m -d=testing=2

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

type hashIface interface {
	Sum() []byte
}

type clonableHashIface interface {
	Sum() []byte
	Clone() hashIface
}

type hash struct{ state [32]byte }

func (h *hash) Sum() []byte { // ERROR "can inline" "h does not escape"
	return make([]byte, 32) // ERROR "escapes"
}

func (h *hash) Clone() hashIface { // ERROR "can inline" "h does not escape"
	c := *h // ERROR "moved to heap: c"
	return &c
}

type hash2 struct{ state [32]byte }

func (h *hash2) Sum() []byte { // ERROR "can inline" "h does not escape"
	return make([]byte, 32) // ERROR "escapes"
}

func (h *hash2) Clone() hashIface { // ERROR "can inline" "h does not escape"
	c := *h // ERROR "moved to heap: c"
	return &c
}

func newHash() hashIface { // ERROR "can inline"
	return &hash{} // ERROR "escapes"
}

func cloneHash1(h hashIface) hashIface { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone()
	}
	return &hash{} // ERROR "escapes"
}

func cloneHash2(h hashIface) hashIface { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone()
	}
	return nil
}

func cloneHash3(h hashIface) hashIface { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone()
	}
	return &hash2{} // ERROR "escapes"
}

func cloneHashWithBool1(h hashIface) (hashIface, bool) { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone(), true
	}
	return &hash{}, false // ERROR "escapes"
}

func cloneHashWithBool2(h hashIface) (hashIface, bool) { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone(), true
	}
	return nil, false
}

func cloneHashWithBool3(h hashIface) (hashIface, bool) { // ERROR "can inline" "leaking param: h"
	if h, ok := h.(clonableHashIface); ok {
		return h.Clone(), true
	}
	return &hash2{}, false // ERROR "escapes"
}

func interleavedWithTypeAssertions() {
	h1 := newHash() // ERROR "&hash{} does not escape" "inlining call"
	_ = h1.Sum()    // ERROR "devirtualizing h1.Sum to \*hash" "inlining call" "make\(\[\]byte, 32\) does not escape"

	h2 := cloneHash1(h1) // ERROR "inlining call to cloneHash1" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone" "&hash{} does not escape"
	_ = h2.Sum()         // ERROR "devirtualizing h2.Sum to \*hash" "inlining call" "make\(\[\]byte, 32\) does not escape"

	h3 := cloneHash2(h1) // ERROR "inlining call to cloneHash2" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone"
	_ = h3.Sum()         // ERROR "devirtualizing h3.Sum to \*hash" "inlining call" "make\(\[\]byte, 32\) does not escape"

	h4 := cloneHash3(h1) // ERROR "inlining call to cloneHash3" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone" "moved to heap: c" "&hash2{} escapes to heap"
	_ = h4.Sum()

	h5, _ := cloneHashWithBool1(h1) // ERROR "inlining call to cloneHashWithBool1" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone" "&hash{} does not escape"
	_ = h5.Sum()                    // ERROR "devirtualizing h5.Sum to \*hash" "inlining call" "make\(\[\]byte, 32\) does not escape"

	h6, _ := cloneHashWithBool2(h1) // ERROR "inlining call to cloneHashWithBool2" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone"
	_ = h6.Sum()                    // ERROR "devirtualizing h6.Sum to \*hash" "inlining call" "make\(\[\]byte, 32\) does not escape"

	h7, _ := cloneHashWithBool3(h1) // ERROR "inlining call to cloneHashWithBool3" "devirtualizing h.Clone to \*hash" "inlining call to \(\*hash\).Clone" "moved to heap: c" "&hash2{} escapes to heap"
	_ = h7.Sum()
}
