// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build icu

package main

/*
#cgo LDFLAGS: -licui18n -licuuc
#include <stdlib.h>
#include <unicode/ucol.h>
#include <unicode/uiter.h>
#include <unicode/utypes.h>
*/
import "C"
import (
	"fmt"
	"log"
	"unicode/utf16"
	"unicode/utf8"
	"unsafe"
)

func init() {
	AddFactory(CollatorFactory{"icu", newUTF16,
		"Main ICU collator, using native strings."})
	AddFactory(CollatorFactory{"icu8", newUTF8iter,
		"ICU collator using ICU iterators to process UTF8."})
	AddFactory(CollatorFactory{"icu16", newUTF8conv,
		"ICU collation by first converting UTF8 to UTF16."})
}

func icuCharP(s []byte) *C.char {
	return (*C.char)(unsafe.Pointer(&s[0]))
}

func icuUInt8P(s []byte) *C.uint8_t {
	return (*C.uint8_t)(unsafe.Pointer(&s[0]))
}

func icuUCharP(s []uint16) *C.UChar {
	return (*C.UChar)(unsafe.Pointer(&s[0]))
}
func icuULen(s []uint16) C.int32_t {
	return C.int32_t(len(s))
}
func icuSLen(s []byte) C.int32_t {
	return C.int32_t(len(s))
}

// icuCollator implements a Collator based on ICU.
type icuCollator struct {
	loc    *C.char
	col    *C.UCollator
	keyBuf []byte
}

const growBufSize = 10 * 1024 * 1024

func (c *icuCollator) init(locale string) error {
	err := C.UErrorCode(0)
	c.loc = C.CString(locale)
	c.col = C.ucol_open(c.loc, &err)
	if err > 0 {
		return fmt.Errorf("failed opening collator for %q", locale)
	} else if err < 0 {
		loc := C.ucol_getLocaleByType(c.col, 0, &err)
		fmt, ok := map[int]string{
			-127: "warning: using default collator: %s",
			-128: "warning: using fallback collator: %s",
		}[int(err)]
		if ok {
			log.Printf(fmt, C.GoString(loc))
		}
	}
	c.keyBuf = make([]byte, 0, growBufSize)
	return nil
}

func (c *icuCollator) buf() (*C.uint8_t, C.int32_t) {
	if len(c.keyBuf) == cap(c.keyBuf) {
		c.keyBuf = make([]byte, 0, growBufSize)
	}
	b := c.keyBuf[len(c.keyBuf):cap(c.keyBuf)]
	return icuUInt8P(b), icuSLen(b)
}

func (c *icuCollator) extendBuf(n C.int32_t) []byte {
	end := len(c.keyBuf) + int(n)
	if end > cap(c.keyBuf) {
		if len(c.keyBuf) == 0 {
			log.Fatalf("icuCollator: max string size exceeded: %v > %v", n, growBufSize)
		}
		c.keyBuf = make([]byte, 0, growBufSize)
		return nil
	}
	b := c.keyBuf[len(c.keyBuf):end]
	c.keyBuf = c.keyBuf[:end]
	return b
}

func (c *icuCollator) Close() error {
	C.ucol_close(c.col)
	C.free(unsafe.Pointer(c.loc))
	return nil
}

// icuUTF16 implements the Collator interface.
type icuUTF16 struct {
	icuCollator
}

func newUTF16(locale string) (Collator, error) {
	c := &icuUTF16{}
	return c, c.init(locale)
}

func (c *icuUTF16) Compare(a, b Input) int {
	return int(C.ucol_strcoll(c.col, icuUCharP(a.UTF16), icuULen(a.UTF16), icuUCharP(b.UTF16), icuULen(b.UTF16)))
}

func (c *icuUTF16) Key(s Input) []byte {
	bp, bn := c.buf()
	n := C.ucol_getSortKey(c.col, icuUCharP(s.UTF16), icuULen(s.UTF16), bp, bn)
	if b := c.extendBuf(n); b != nil {
		return b
	}
	return c.Key(s)
}

// icuUTF8iter implements the Collator interface
// This implementation wraps the UTF8 string in an iterator
// which is passed to the collator.
type icuUTF8iter struct {
	icuCollator
	a, b C.UCharIterator
}

func newUTF8iter(locale string) (Collator, error) {
	c := &icuUTF8iter{}
	return c, c.init(locale)
}

func (c *icuUTF8iter) Compare(a, b Input) int {
	err := C.UErrorCode(0)
	C.uiter_setUTF8(&c.a, icuCharP(a.UTF8), icuSLen(a.UTF8))
	C.uiter_setUTF8(&c.b, icuCharP(b.UTF8), icuSLen(b.UTF8))
	return int(C.ucol_strcollIter(c.col, &c.a, &c.b, &err))
}

func (c *icuUTF8iter) Key(s Input) []byte {
	err := C.UErrorCode(0)
	state := [2]C.uint32_t{}
	C.uiter_setUTF8(&c.a, icuCharP(s.UTF8), icuSLen(s.UTF8))
	bp, bn := c.buf()
	n := C.ucol_nextSortKeyPart(c.col, &c.a, &(state[0]), bp, bn, &err)
	if n >= bn {
		// Force failure.
		if c.extendBuf(n+1) != nil {
			log.Fatal("expected extension to fail")
		}
		return c.Key(s)
	}
	return c.extendBuf(n)
}

// icuUTF8conv implementes the Collator interface.
// This implentation first converts the give UTF8 string
// to UTF16 and then calls the main ICU collation function.
type icuUTF8conv struct {
	icuCollator
}

func newUTF8conv(locale string) (Collator, error) {
	c := &icuUTF8conv{}
	return c, c.init(locale)
}

func (c *icuUTF8conv) Compare(sa, sb Input) int {
	a := encodeUTF16(sa.UTF8)
	b := encodeUTF16(sb.UTF8)
	return int(C.ucol_strcoll(c.col, icuUCharP(a), icuULen(a), icuUCharP(b), icuULen(b)))
}

func (c *icuUTF8conv) Key(s Input) []byte {
	a := encodeUTF16(s.UTF8)
	bp, bn := c.buf()
	n := C.ucol_getSortKey(c.col, icuUCharP(a), icuULen(a), bp, bn)
	if b := c.extendBuf(n); b != nil {
		return b
	}
	return c.Key(s)
}

func encodeUTF16(b []byte) []uint16 {
	a := []uint16{}
	for len(b) > 0 {
		r, sz := utf8.DecodeRune(b)
		b = b[sz:]
		r1, r2 := utf16.EncodeRune(r)
		if r1 != 0xFFFD {
			a = append(a, uint16(r1), uint16(r2))
		} else {
			a = append(a, uint16(r))
		}
	}
	return a
}
