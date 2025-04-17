// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package windows

import "syscall"

// NTUnicodeString is a UTF-16 string for NT native APIs, corresponding to UNICODE_STRING.
type NTUnicodeString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        *uint16
}

// NewNTUnicodeString returns a new NTUnicodeString structure for use with native
// NT APIs that work over the NTUnicodeString type. Note that most Windows APIs
// do not use NTUnicodeString, and instead UTF16PtrFromString should be used for
// the more common *uint16 string type.
func NewNTUnicodeString(s string) (*NTUnicodeString, error) {
	s16, err := syscall.UTF16FromString(s)
	if err != nil {
		return nil, err
	}
	n := uint16(len(s16) * 2)
	// https://learn.microsoft.com/en-us/windows-hardware/drivers/ddi/wdmsec/nf-wdmsec-wdmlibrtlinitunicodestringex
	return &NTUnicodeString{
		Length:        n - 2, // subtract 2 bytes for the NUL terminator
		MaximumLength: n,
		Buffer:        &s16[0],
	}, nil
}
