// errorcheck -0 -m

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

type hashIface interface {
	Sum() []byte
}

type cloneableHashIface interface {
	hashIface
	Clone() hashIface
}

type hash struct{ state [32]byte }

func (h *hash) Sum() []byte { // ERROR "can inline \(\*hash\).Sum$" "h does not escape$"
	return make([]byte, 32) // ERROR "make\(\[\]byte, 32\) escapes to heap$"
}

func (h *hash) Clone() hashIface { // ERROR "can inline \(\*hash\).Clone$" "h does not escape$"
	c := *h // ERROR "moved to heap: c$"
	return &c
}

type hash2 struct{ state [32]byte }

func (h *hash2) Sum() []byte { // ERROR "can inline \(\*hash2\).Sum$" "h does not escape$"
	return make([]byte, 32) // ERROR "make\(\[\]byte, 32\) escapes to heap$"
}

func (h *hash2) Clone() hashIface { // ERROR "can inline \(\*hash2\).Clone$" "h does not escape$"
	c := *h // ERROR "moved to heap: c$"
	return &c
}

func newHash() hashIface { // ERROR "can inline newHash$"
	return &hash{} // ERROR "&hash{} escapes to heap$"
}

func cloneHash1(h hashIface) hashIface { // ERROR "can inline cloneHash1$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone()
	}
	return &hash{} // ERROR "&hash{} escapes to heap$"
}

func cloneHash2(h hashIface) hashIface { // ERROR "can inline cloneHash2$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone()
	}
	return nil
}

func cloneHash3(h hashIface) hashIface { // ERROR "can inline cloneHash3$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone()
	}
	return &hash2{} // ERROR "&hash2{} escapes to heap$"
}

func cloneHashWithBool1(h hashIface) (hashIface, bool) { // ERROR "can inline cloneHashWithBool1$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone(), true
	}
	return &hash{}, false // ERROR "&hash{} escapes to heap$"
}

func cloneHashWithBool2(h hashIface) (hashIface, bool) { // ERROR "can inline cloneHashWithBool2$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone(), true
	}
	return nil, false
}

func cloneHashWithBool3(h hashIface) (hashIface, bool) { // ERROR "can inline cloneHashWithBool3$" "leaking param: h$"
	if h, ok := h.(cloneableHashIface); ok {
		return h.Clone(), true
	}
	return &hash2{}, false // ERROR "&hash2{} escapes to heap$"
}

func interleavedWithTypeAssertions() {
	h1 := newHash() // ERROR "&hash{} does not escape$" "inlining call to newHash"
	_ = h1.Sum()    // ERROR "devirtualizing h1.Sum to \*hash$" "inlining call to \(\*hash\).Sum" "make\(\[\]byte, 32\) does not escape$"

	h2 := cloneHash1(h1) // ERROR "&hash{} does not escape$" "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHash1"
	_ = h2.Sum()         // ERROR "devirtualizing h2.Sum to \*hash$" "inlining call to \(\*hash\).Sum" "make\(\[\]byte, 32\) does not escape$"

	h3 := cloneHash2(h1) // ERROR "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHash2"
	_ = h3.Sum()         // ERROR "devirtualizing h3.Sum to \*hash$" "inlining call to \(\*hash\).Sum" "make\(\[\]byte, 32\) does not escape$"

	h4 := cloneHash3(h1) // ERROR "&hash2{} escapes to heap$" "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHash3" "moved to heap: c$"
	_ = h4.Sum()

	h5, _ := cloneHashWithBool1(h1) // ERROR "&hash{} does not escape$" "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHashWithBool1"
	_ = h5.Sum()                    // ERROR "devirtualizing h5.Sum to \*hash$" "inlining call to \(\*hash\).Sum" "make\(\[\]byte, 32\) does not escape$"

	h6, _ := cloneHashWithBool2(h1) // ERROR "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHashWithBool2"
	_ = h6.Sum()                    // ERROR "devirtualizing h6.Sum to \*hash$" "inlining call to \(\*hash\).Sum" "make\(\[\]byte, 32\) does not escape$"

	h7, _ := cloneHashWithBool3(h1) // ERROR "&hash2{} escapes to heap$" "devirtualizing h.Clone to \*hash$" "inlining call to \(\*hash\).Clone" "inlining call to cloneHashWithBool3" "moved to heap: c$"
	_ = h7.Sum()
}

type cloneableHashError interface {
	hashIface
	Clone() (hashIface, error)
}

type hash3 struct{ state [32]byte }

func newHash3() hashIface { // ERROR "can inline newHash3$"
	return &hash3{} // ERROR "&hash3{} escapes to heap$"
}

func (h *hash3) Sum() []byte { // ERROR "can inline \(\*hash3\).Sum$" "h does not escape$"
	return make([]byte, 32) // ERROR "make\(\[\]byte, 32\) escapes to heap$"
}

func (h *hash3) Clone() (hashIface, error) { // ERROR "can inline \(\*hash3\).Clone$" "h does not escape$"
	c := *h // ERROR "moved to heap: c$"
	return &c, nil
}

func interleavedCloneableHashError() {
	h1 := newHash3() // ERROR "&hash3{} does not escape$" "inlining call to newHash3"
	_ = h1.Sum()     // ERROR "devirtualizing h1.Sum to \*hash3$" "inlining call to \(\*hash3\).Sum" "make\(\[\]byte, 32\) does not escape$"

	if h1, ok := h1.(cloneableHashError); ok {
		h2, err := h1.Clone() // ERROR "devirtualizing h1.Clone to \*hash3$" "inlining call to \(\*hash3\).Clone"
		if err == nil {
			_ = h2.Sum() // ERROR "devirtualizing h2.Sum to \*hash3$" "inlining call to \(\*hash3\).Sum" "make\(\[\]byte, 32\) does not escape$"
		}
	}
}
