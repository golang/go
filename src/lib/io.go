// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io
import os "os"

export func StringToBytes(b *[]byte, s string) bool {
	if len(s) >= len(b) {
		return false
	}
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	b[len(s)] = '\000';	// not necessary - memory is zeroed - but be explicit
	return true
}

export type Read interface {
	Read(p *[]byte) (n int, err *os.Error);
}

export type Write interface {
	Write(p *[]byte) (n int, err *os.Error);
}

export func WriteString(w Write, s string) (n int, err *os.Error) {
	b := new([]byte, len(s)+1)
	if !StringToBytes(b, s) {
		return -1, os.EINVAL
	}
	// BUG return w.Write(b[0:len(s)])
	r, e := w.Write(b[0:len(s)])
	return r, e
}

