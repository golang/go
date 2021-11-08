// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || windows

package poll_test

import (
	. "internal/poll"
	"io"
	"testing"
)

var eofErrorTests = []struct {
	n        int
	err      error
	fd       *FD
	expected error
}{
	{100, nil, &FD{ZeroReadIsEOF: true}, nil},
	{100, io.EOF, &FD{ZeroReadIsEOF: true}, io.EOF},
	{100, ErrNetClosing, &FD{ZeroReadIsEOF: true}, ErrNetClosing},
	{0, nil, &FD{ZeroReadIsEOF: true}, io.EOF},
	{0, io.EOF, &FD{ZeroReadIsEOF: true}, io.EOF},
	{0, ErrNetClosing, &FD{ZeroReadIsEOF: true}, ErrNetClosing},

	{100, nil, &FD{ZeroReadIsEOF: false}, nil},
	{100, io.EOF, &FD{ZeroReadIsEOF: false}, io.EOF},
	{100, ErrNetClosing, &FD{ZeroReadIsEOF: false}, ErrNetClosing},
	{0, nil, &FD{ZeroReadIsEOF: false}, nil},
	{0, io.EOF, &FD{ZeroReadIsEOF: false}, io.EOF},
	{0, ErrNetClosing, &FD{ZeroReadIsEOF: false}, ErrNetClosing},
}

func TestEOFError(t *testing.T) {
	for _, tt := range eofErrorTests {
		actual := tt.fd.EOFError(tt.n, tt.err)
		if actual != tt.expected {
			t.Errorf("eofError(%v, %v, %v): expected %v, actual %v", tt.n, tt.err, tt.fd.ZeroReadIsEOF, tt.expected, actual)
		}
	}
}
