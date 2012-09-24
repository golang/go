// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin

package main

/*
#cgo LDFLAGS: -framework CoreFoundation
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CoreFoundation.h>
*/
import "C"
import (
	"unsafe"
)

func init() {
	AddFactory(CollatorFactory{"osx", newOSX16Collator,
		"OS X/Darwin collator, using native strings."})
	AddFactory(CollatorFactory{"osx8", newOSX8Collator,
		"OS X/Darwin collator for UTF-8."})
}

func osxUInt8P(s []byte) *C.UInt8 {
	return (*C.UInt8)(unsafe.Pointer(&s[0]))
}

func osxCharP(s []uint16) *C.UniChar {
	return (*C.UniChar)(unsafe.Pointer(&s[0]))
}

// osxCollator implements an Collator based on OS X's CoreFoundation.
type osxCollator struct {
	loc C.CFLocaleRef
	opt C.CFStringCompareFlags
}

func (c *osxCollator) init(locale string) {
	l := C.CFStringCreateWithBytes(
		nil,
		osxUInt8P([]byte(locale)),
		C.CFIndex(len(locale)),
		C.kCFStringEncodingUTF8,
		C.Boolean(0),
	)
	c.loc = C.CFLocaleCreate(nil, l)
}

func newOSX8Collator(locale string) (Collator, error) {
	c := &osx8Collator{}
	c.init(locale)
	return c, nil
}

func newOSX16Collator(locale string) (Collator, error) {
	c := &osx16Collator{}
	c.init(locale)
	return c, nil
}

func (c osxCollator) Key(s Input) []byte {
	return nil // sort keys not supported by OS X CoreFoundation
}

type osx8Collator struct {
	osxCollator
}

type osx16Collator struct {
	osxCollator
}

func (c osx16Collator) Compare(a, b Input) int {
	sa := C.CFStringCreateWithCharactersNoCopy(
		nil,
		osxCharP(a.UTF16),
		C.CFIndex(len(a.UTF16)),
		C.kCFAllocatorNull,
	)
	sb := C.CFStringCreateWithCharactersNoCopy(
		nil,
		osxCharP(b.UTF16),
		C.CFIndex(len(b.UTF16)),
		C.kCFAllocatorNull,
	)
	_range := C.CFRangeMake(0, C.CFStringGetLength(sa))
	return int(C.CFStringCompareWithOptionsAndLocale(sa, sb, _range, c.opt, c.loc))
}

func (c osx8Collator) Compare(a, b Input) int {
	sa := C.CFStringCreateWithBytesNoCopy(
		nil,
		osxUInt8P(a.UTF8),
		C.CFIndex(len(a.UTF8)),
		C.kCFStringEncodingUTF8,
		C.Boolean(0),
		C.kCFAllocatorNull,
	)
	sb := C.CFStringCreateWithBytesNoCopy(
		nil,
		osxUInt8P(b.UTF8),
		C.CFIndex(len(b.UTF8)),
		C.kCFStringEncodingUTF8,
		C.Boolean(0),
		C.kCFAllocatorNull,
	)
	_range := C.CFRangeMake(0, C.CFStringGetLength(sa))
	return int(C.CFStringCompareWithOptionsAndLocale(sa, sb, _range, c.opt, c.loc))
}
