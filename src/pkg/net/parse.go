// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple file i/o and string manipulation, to avoid
// depending on strconv and bufio and strings.

package net

import (
	"io"
	"os"
)

type file struct {
	file  *os.File
	data  []byte
	atEOF bool
}

func (f *file) close() { f.file.Close() }

func (f *file) getLineFromData() (s string, ok bool) {
	data := f.data
	i := 0
	for i = 0; i < len(data); i++ {
		if data[i] == '\n' {
			s = string(data[0:i])
			ok = true
			// move data
			i++
			n := len(data) - i
			copy(data[0:], data[i:])
			f.data = data[0:n]
			return
		}
	}
	if f.atEOF && len(f.data) > 0 {
		// EOF, return all we have
		s = string(data)
		f.data = f.data[0:0]
		ok = true
	}
	return
}

func (f *file) readLine() (s string, ok bool) {
	if s, ok = f.getLineFromData(); ok {
		return
	}
	if len(f.data) < cap(f.data) {
		ln := len(f.data)
		n, err := io.ReadFull(f.file, f.data[ln:cap(f.data)])
		if n >= 0 {
			f.data = f.data[0 : ln+n]
		}
		if err == os.EOF {
			f.atEOF = true
		}
	}
	s, ok = f.getLineFromData()
	return
}

func open(name string) (*file, os.Error) {
	fd, err := os.Open(name, os.O_RDONLY, 0)
	if err != nil {
		return nil, err
	}
	return &file{fd, make([]byte, 1024)[0:0], false}, nil
}

func byteIndex(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}

// Count occurrences in s of any bytes in t.
func countAnyByte(s string, t string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if byteIndex(t, s[i]) >= 0 {
			n++
		}
	}
	return n
}

// Split s at any bytes in t.
func splitAtBytes(s string, t string) []string {
	a := make([]string, 1+countAnyByte(s, t))
	n := 0
	last := 0
	for i := 0; i < len(s); i++ {
		if byteIndex(t, s[i]) >= 0 {
			if last < i {
				a[n] = string(s[last:i])
				n++
			}
			last = i + 1
		}
	}
	if last < len(s) {
		a[n] = string(s[last:])
		n++
	}
	return a[0:n]
}

func getFields(s string) []string { return splitAtBytes(s, " \r\t\n") }

// Bigger than we need, not too big to worry about overflow
const big = 0xFFFFFF

// Decimal to integer starting at &s[i0].
// Returns number, new offset, success.
func dtoi(s string, i0 int) (n int, i int, ok bool) {
	n = 0
	for i = i0; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		n = n*10 + int(s[i]-'0')
		if n >= big {
			return 0, i, false
		}
	}
	if i == i0 {
		return 0, i, false
	}
	return n, i, true
}

// Hexadecimal to integer starting at &s[i0].
// Returns number, new offset, success.
func xtoi(s string, i0 int) (n int, i int, ok bool) {
	n = 0
	for i = i0; i < len(s); i++ {
		if '0' <= s[i] && s[i] <= '9' {
			n *= 16
			n += int(s[i] - '0')
		} else if 'a' <= s[i] && s[i] <= 'f' {
			n *= 16
			n += int(s[i]-'a') + 10
		} else if 'A' <= s[i] && s[i] <= 'F' {
			n *= 16
			n += int(s[i]-'A') + 10
		} else {
			break
		}
		if n >= big {
			return 0, i, false
		}
	}
	if i == i0 {
		return 0, i, false
	}
	return n, i, true
}

// Integer to decimal.
func itoa(i int) string {
	var buf [30]byte
	n := len(buf)
	neg := false
	if i < 0 {
		i = -i
		neg = true
	}
	ui := uint(i)
	for ui > 0 || n == len(buf) {
		n--
		buf[n] = byte('0' + ui%10)
		ui /= 10
	}
	if neg {
		n--
		buf[n] = '-'
	}
	return string(buf[n:])
}

// Number of occurrences of b in s.
func count(s string, b byte) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			n++
		}
	}
	return n
}

// Returns the prefix of s up to but not including the character c
func prefixBefore(s string, c byte) string {
	for i, v := range s {
		if v == int(c) {
			return s[0:i]
		}
	}
	return s
}

// Index of rightmost occurrence of b in s.
func last(s string, b byte) int {
	i := len(s)
	for i--; i >= 0; i-- {
		if s[i] == b {
			break
		}
	}
	return i
}
