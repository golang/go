// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Simple file i/o and string manipulation, to avoid
// depending on strconv and bufio and strings.

package net

import (
	"internal/bytealg"
	"io"
	"os"
	"time"
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
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			f.atEOF = true
		}
	}
	s, ok = f.getLineFromData()
	return
}

func open(name string) (*file, error) {
	fd, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	return &file{fd, make([]byte, 0, 64*1024), false}, nil
}

func stat(name string) (mtime time.Time, size int64, err error) {
	st, err := os.Stat(name)
	if err != nil {
		return time.Time{}, 0, err
	}
	return st.ModTime(), st.Size(), nil
}

// Count occurrences in s of any bytes in t.
func countAnyByte(s string, t string) int {
	n := 0
	for i := 0; i < len(s); i++ {
		if bytealg.IndexByteString(t, s[i]) >= 0 {
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
		if bytealg.IndexByteString(t, s[i]) >= 0 {
			if last < i {
				a[n] = s[last:i]
				n++
			}
			last = i + 1
		}
	}
	if last < len(s) {
		a[n] = s[last:]
		n++
	}
	return a[0:n]
}

func getFields(s string) []string { return splitAtBytes(s, " \r\t\n") }

// Bigger than we need, not too big to worry about overflow
const big = 0xFFFFFF

// Decimal to integer.
// Returns number, characters consumed, success.
func dtoi(s string) (n int, i int, ok bool) {
	n = 0
	for i = 0; i < len(s) && '0' <= s[i] && s[i] <= '9'; i++ {
		n = n*10 + int(s[i]-'0')
		if n >= big {
			return big, i, false
		}
	}
	if i == 0 {
		return 0, 0, false
	}
	return n, i, true
}

// Hexadecimal to integer.
// Returns number, characters consumed, success.
func xtoi(s string) (n int, i int, ok bool) {
	n = 0
	for i = 0; i < len(s); i++ {
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
	if i == 0 {
		return 0, i, false
	}
	return n, i, true
}

// xtoi2 converts the next two hex digits of s into a byte.
// If s is longer than 2 bytes then the third byte must be e.
// If the first two bytes of s are not hex digits or the third byte
// does not match e, false is returned.
func xtoi2(s string, e byte) (byte, bool) {
	if len(s) > 2 && s[2] != e {
		return 0, false
	}
	n, ei, ok := xtoi(s[:2])
	return byte(n), ok && ei == 2
}

// Convert i to a hexadecimal string. Leading zeros are not printed.
func appendHex(dst []byte, i uint32) []byte {
	if i == 0 {
		return append(dst, '0')
	}
	for j := 7; j >= 0; j-- {
		v := i >> uint(j*4)
		if v > 0 {
			dst = append(dst, hexDigit[v&0xf])
		}
	}
	return dst
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

// hasUpperCase tells whether the given string contains at least one upper-case.
func hasUpperCase(s string) bool {
	for i := range s {
		if 'A' <= s[i] && s[i] <= 'Z' {
			return true
		}
	}
	return false
}

// lowerASCIIBytes makes x ASCII lowercase in-place.
func lowerASCIIBytes(x []byte) {
	for i, b := range x {
		if 'A' <= b && b <= 'Z' {
			x[i] += 'a' - 'A'
		}
	}
}

// lowerASCII returns the ASCII lowercase version of b.
func lowerASCII(b byte) byte {
	if 'A' <= b && b <= 'Z' {
		return b + ('a' - 'A')
	}
	return b
}

// trimSpace returns x without any leading or trailing ASCII whitespace.
func trimSpace(x []byte) []byte {
	for len(x) > 0 && isSpace(x[0]) {
		x = x[1:]
	}
	for len(x) > 0 && isSpace(x[len(x)-1]) {
		x = x[:len(x)-1]
	}
	return x
}

// isSpace reports whether b is an ASCII space character.
func isSpace(b byte) bool {
	return b == ' ' || b == '\t' || b == '\n' || b == '\r'
}

// removeComment returns line, removing any '#' byte and any following
// bytes.
func removeComment(line []byte) []byte {
	if i := bytealg.IndexByte(line, '#'); i != -1 {
		return line[:i]
	}
	return line
}

// foreachLine runs fn on each line of x.
// Each line (except for possibly the last) ends in '\n'.
// It returns the first non-nil error returned by fn.
func foreachLine(x []byte, fn func(line []byte) error) error {
	for len(x) > 0 {
		nl := bytealg.IndexByte(x, '\n')
		if nl == -1 {
			return fn(x)
		}
		line := x[:nl+1]
		x = x[nl+1:]
		if err := fn(line); err != nil {
			return err
		}
	}
	return nil
}

// foreachField runs fn on each non-empty run of non-space bytes in x.
// It returns the first non-nil error returned by fn.
func foreachField(x []byte, fn func(field []byte) error) error {
	x = trimSpace(x)
	for len(x) > 0 {
		sp := bytealg.IndexByte(x, ' ')
		if sp == -1 {
			return fn(x)
		}
		if field := trimSpace(x[:sp]); len(field) > 0 {
			if err := fn(field); err != nil {
				return err
			}
		}
		x = trimSpace(x[sp+1:])
	}
	return nil
}

// stringsHasSuffix is strings.HasSuffix. It reports whether s ends in
// suffix.
func stringsHasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// stringsHasSuffixFold reports whether s ends in suffix,
// ASCII-case-insensitively.
func stringsHasSuffixFold(s, suffix string) bool {
	return len(s) >= len(suffix) && stringsEqualFold(s[len(s)-len(suffix):], suffix)
}

// stringsHasPrefix is strings.HasPrefix. It reports whether s begins with prefix.
func stringsHasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

// stringsEqualFold is strings.EqualFold, ASCII only. It reports whether s and t
// are equal, ASCII-case-insensitively.
func stringsEqualFold(s, t string) bool {
	if len(s) != len(t) {
		return false
	}
	for i := 0; i < len(s); i++ {
		if lowerASCII(s[i]) != lowerASCII(t[i]) {
			return false
		}
	}
	return true
}

func readFull(r io.Reader) (all []byte, err error) {
	buf := make([]byte, 1024)
	for {
		n, err := r.Read(buf)
		all = append(all, buf[:n]...)
		if err == io.EOF {
			return all, nil
		}
		if err != nil {
			return nil, err
		}
	}
}
