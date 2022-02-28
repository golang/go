// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"fmt"
	"strconv"
	"strings"
	"time"
)

// hasNUL reports whether the NUL character exists within s.
func hasNUL(s string) bool {
	return strings.Contains(s, "\x00")
}

// isASCII reports whether the input is an ASCII C-style string.
func isASCII(s string) bool {
	for _, c := range s {
		if c >= 0x80 || c == 0x00 {
			return false
		}
	}
	return true
}

// toASCII converts the input to an ASCII C-style string.
// This is a best effort conversion, so invalid characters are dropped.
func toASCII(s string) string {
	if isASCII(s) {
		return s
	}
	b := make([]byte, 0, len(s))
	for _, c := range s {
		if c < 0x80 && c != 0x00 {
			b = append(b, byte(c))
		}
	}
	return string(b)
}

type parser struct {
	err error // Last error seen
}

type formatter struct {
	err error // Last error seen
}

// parseString parses bytes as a NUL-terminated C-style string.
// If a NUL byte is not found then the whole slice is returned as a string.
func (*parser) parseString(b []byte) string {
	if i := bytes.IndexByte(b, 0); i >= 0 {
		return string(b[:i])
	}
	return string(b)
}

// formatString copies s into b, NUL-terminating if possible.
func (f *formatter) formatString(b []byte, s string) {
	if len(s) > len(b) {
		f.err = ErrFieldTooLong
	}
	copy(b, s)
	if len(s) < len(b) {
		b[len(s)] = 0
	}

	// Some buggy readers treat regular files with a trailing slash
	// in the V7 path field as a directory even though the full path
	// recorded elsewhere (e.g., via PAX record) contains no trailing slash.
	if len(s) > len(b) && b[len(b)-1] == '/' {
		n := len(strings.TrimRight(s[:len(b)], "/"))
		b[n] = 0 // Replace trailing slash with NUL terminator
	}
}

// fitsInBase256 reports whether x can be encoded into n bytes using base-256
// encoding. Unlike octal encoding, base-256 encoding does not require that the
// string ends with a NUL character. Thus, all n bytes are available for output.
//
// If operating in binary mode, this assumes strict GNU binary mode; which means
// that the first byte can only be either 0x80 or 0xff. Thus, the first byte is
// equivalent to the sign bit in two's complement form.
func fitsInBase256(n int, x int64) bool {
	binBits := uint(n-1) * 8
	return n >= 9 || (x >= -1<<binBits && x < 1<<binBits)
}

// parseNumeric parses the input as being encoded in either base-256 or octal.
// This function may return negative numbers.
// If parsing fails or an integer overflow occurs, err will be set.
func (p *parser) parseNumeric(b []byte) int64 {
	// Check for base-256 (binary) format first.
	// If the first bit is set, then all following bits constitute a two's
	// complement encoded number in big-endian byte order.
	if len(b) > 0 && b[0]&0x80 != 0 {
		// Handling negative numbers relies on the following identity:
		//	-a-1 == ^a
		//
		// If the number is negative, we use an inversion mask to invert the
		// data bytes and treat the value as an unsigned number.
		var inv byte // 0x00 if positive or zero, 0xff if negative
		if b[0]&0x40 != 0 {
			inv = 0xff
		}

		var x uint64
		for i, c := range b {
			c ^= inv // Inverts c only if inv is 0xff, otherwise does nothing
			if i == 0 {
				c &= 0x7f // Ignore signal bit in first byte
			}
			if (x >> 56) > 0 {
				p.err = ErrHeader // Integer overflow
				return 0
			}
			x = x<<8 | uint64(c)
		}
		if (x >> 63) > 0 {
			p.err = ErrHeader // Integer overflow
			return 0
		}
		if inv == 0xff {
			return ^int64(x)
		}
		return int64(x)
	}

	// Normal case is base-8 (octal) format.
	return p.parseOctal(b)
}

// formatNumeric encodes x into b using base-8 (octal) encoding if possible.
// Otherwise it will attempt to use base-256 (binary) encoding.
func (f *formatter) formatNumeric(b []byte, x int64) {
	if fitsInOctal(len(b), x) {
		f.formatOctal(b, x)
		return
	}

	if fitsInBase256(len(b), x) {
		for i := len(b) - 1; i >= 0; i-- {
			b[i] = byte(x)
			x >>= 8
		}
		b[0] |= 0x80 // Highest bit indicates binary format
		return
	}

	f.formatOctal(b, 0) // Last resort, just write zero
	f.err = ErrFieldTooLong
}

func (p *parser) parseOctal(b []byte) int64 {
	// Because unused fields are filled with NULs, we need
	// to skip leading NULs. Fields may also be padded with
	// spaces or NULs.
	// So we remove leading and trailing NULs and spaces to
	// be sure.
	b = bytes.Trim(b, " \x00")

	if len(b) == 0 {
		return 0
	}
	x, perr := strconv.ParseUint(p.parseString(b), 8, 64)
	if perr != nil {
		p.err = ErrHeader
	}
	return int64(x)
}

func (f *formatter) formatOctal(b []byte, x int64) {
	if !fitsInOctal(len(b), x) {
		x = 0 // Last resort, just write zero
		f.err = ErrFieldTooLong
	}

	s := strconv.FormatInt(x, 8)
	// Add leading zeros, but leave room for a NUL.
	if n := len(b) - len(s) - 1; n > 0 {
		s = strings.Repeat("0", n) + s
	}
	f.formatString(b, s)
}

// fitsInOctal reports whether the integer x fits in a field n-bytes long
// using octal encoding with the appropriate NUL terminator.
func fitsInOctal(n int, x int64) bool {
	octBits := uint(n-1) * 3
	return x >= 0 && (n >= 22 || x < 1<<octBits)
}

// parsePAXTime takes a string of the form %d.%d as described in the PAX
// specification. Note that this implementation allows for negative timestamps,
// which is allowed for by the PAX specification, but not always portable.
func parsePAXTime(s string) (time.Time, error) {
	const maxNanoSecondDigits = 9

	// Split string into seconds and sub-seconds parts.
	ss, sn, _ := strings.Cut(s, ".")

	// Parse the seconds.
	secs, err := strconv.ParseInt(ss, 10, 64)
	if err != nil {
		return time.Time{}, ErrHeader
	}
	if len(sn) == 0 {
		return time.Unix(secs, 0), nil // No sub-second values
	}

	// Parse the nanoseconds.
	if strings.Trim(sn, "0123456789") != "" {
		return time.Time{}, ErrHeader
	}
	if len(sn) < maxNanoSecondDigits {
		sn += strings.Repeat("0", maxNanoSecondDigits-len(sn)) // Right pad
	} else {
		sn = sn[:maxNanoSecondDigits] // Right truncate
	}
	nsecs, _ := strconv.ParseInt(sn, 10, 64) // Must succeed
	if len(ss) > 0 && ss[0] == '-' {
		return time.Unix(secs, -1*nsecs), nil // Negative correction
	}
	return time.Unix(secs, nsecs), nil
}

// formatPAXTime converts ts into a time of the form %d.%d as described in the
// PAX specification. This function is capable of negative timestamps.
func formatPAXTime(ts time.Time) (s string) {
	secs, nsecs := ts.Unix(), ts.Nanosecond()
	if nsecs == 0 {
		return strconv.FormatInt(secs, 10)
	}

	// If seconds is negative, then perform correction.
	sign := ""
	if secs < 0 {
		sign = "-"             // Remember sign
		secs = -(secs + 1)     // Add a second to secs
		nsecs = -(nsecs - 1e9) // Take that second away from nsecs
	}
	return strings.TrimRight(fmt.Sprintf("%s%d.%09d", sign, secs, nsecs), "0")
}

// parsePAXRecord parses the input PAX record string into a key-value pair.
// If parsing is successful, it will slice off the currently read record and
// return the remainder as r.
func parsePAXRecord(s string) (k, v, r string, err error) {
	// The size field ends at the first space.
	nStr, rest, ok := strings.Cut(s, " ")
	if !ok {
		return "", "", s, ErrHeader
	}

	// Parse the first token as a decimal integer.
	n, perr := strconv.ParseInt(nStr, 10, 0) // Intentionally parse as native int
	if perr != nil || n < 5 || n > int64(len(s)) {
		return "", "", s, ErrHeader
	}
	n -= int64(len(nStr) + 1) // convert from index in s to index in rest
	if n <= 0 {
		return "", "", s, ErrHeader
	}

	// Extract everything between the space and the final newline.
	rec, nl, rem := rest[:n-1], rest[n-1:n], rest[n:]
	if nl != "\n" {
		return "", "", s, ErrHeader
	}

	// The first equals separates the key from the value.
	k, v, ok = strings.Cut(rec, "=")
	if !ok {
		return "", "", s, ErrHeader
	}

	if !validPAXRecord(k, v) {
		return "", "", s, ErrHeader
	}
	return k, v, rem, nil
}

// formatPAXRecord formats a single PAX record, prefixing it with the
// appropriate length.
func formatPAXRecord(k, v string) (string, error) {
	if !validPAXRecord(k, v) {
		return "", ErrHeader
	}

	const padding = 3 // Extra padding for ' ', '=', and '\n'
	size := len(k) + len(v) + padding
	size += len(strconv.Itoa(size))
	record := strconv.Itoa(size) + " " + k + "=" + v + "\n"

	// Final adjustment if adding size field increased the record size.
	if len(record) != size {
		size = len(record)
		record = strconv.Itoa(size) + " " + k + "=" + v + "\n"
	}
	return record, nil
}

// validPAXRecord reports whether the key-value pair is valid where each
// record is formatted as:
//	"%d %s=%s\n" % (size, key, value)
//
// Keys and values should be UTF-8, but the number of bad writers out there
// forces us to be a more liberal.
// Thus, we only reject all keys with NUL, and only reject NULs in values
// for the PAX version of the USTAR string fields.
// The key must not contain an '=' character.
func validPAXRecord(k, v string) bool {
	if k == "" || strings.Contains(k, "=") {
		return false
	}
	switch k {
	case paxPath, paxLinkpath, paxUname, paxGname:
		return !hasNUL(v)
	default:
		return !hasNUL(k)
	}
}
