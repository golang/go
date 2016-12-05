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

func isASCII(s string) bool {
	for _, c := range s {
		if c >= 0x80 {
			return false
		}
	}
	return true
}

func toASCII(s string) string {
	if isASCII(s) {
		return s
	}
	var buf bytes.Buffer
	for _, c := range s {
		if c < 0x80 {
			buf.WriteByte(byte(c))
		}
	}
	return buf.String()
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
	n := 0
	for n < len(b) && b[n] != 0 {
		n++
	}
	return string(b[0:n])
}

// Write s into b, terminating it with a NUL if there is room.
func (f *formatter) formatString(b []byte, s string) {
	if len(s) > len(b) {
		f.err = ErrFieldTooLong
		return
	}
	ascii := toASCII(s)
	copy(b, ascii)
	if len(ascii) < len(b) {
		b[len(ascii)] = 0
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
	var binBits = uint(n-1) * 8
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

// Write x into b, as binary (GNUtar/star extension).
func (f *formatter) formatNumeric(b []byte, x int64) {
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
	s := strconv.FormatInt(x, 8)
	// Add leading zeros, but leave room for a NUL.
	if n := len(b) - len(s) - 1; n > 0 {
		s = strings.Repeat("0", n) + s
	}
	f.formatString(b, s)
}

// parsePAXTime takes a string of the form %d.%d as described in the PAX
// specification. Note that this implementation allows for negative timestamps,
// which is allowed for by the PAX specification, but not always portable.
func parsePAXTime(s string) (time.Time, error) {
	const maxNanoSecondDigits = 9

	// Split string into seconds and sub-seconds parts.
	ss, sn := s, ""
	if pos := strings.IndexByte(s, '.'); pos >= 0 {
		ss, sn = s[:pos], s[pos+1:]
	}

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
		return time.Unix(secs, -1*int64(nsecs)), nil // Negative correction
	}
	return time.Unix(secs, int64(nsecs)), nil
}

// TODO(dsnet): Implement formatPAXTime.

// parsePAXRecord parses the input PAX record string into a key-value pair.
// If parsing is successful, it will slice off the currently read record and
// return the remainder as r.
//
// A PAX record is of the following form:
//	"%d %s=%s\n" % (size, key, value)
func parsePAXRecord(s string) (k, v, r string, err error) {
	// The size field ends at the first space.
	sp := strings.IndexByte(s, ' ')
	if sp == -1 {
		return "", "", s, ErrHeader
	}

	// Parse the first token as a decimal integer.
	n, perr := strconv.ParseInt(s[:sp], 10, 0) // Intentionally parse as native int
	if perr != nil || n < 5 || int64(len(s)) < n {
		return "", "", s, ErrHeader
	}

	// Extract everything between the space and the final newline.
	rec, nl, rem := s[sp+1:n-1], s[n-1:n], s[n:]
	if nl != "\n" {
		return "", "", s, ErrHeader
	}

	// The first equals separates the key from the value.
	eq := strings.IndexByte(rec, '=')
	if eq == -1 {
		return "", "", s, ErrHeader
	}
	return rec[:eq], rec[eq+1:], rem, nil
}

// formatPAXRecord formats a single PAX record, prefixing it with the
// appropriate length.
func formatPAXRecord(k, v string) string {
	const padding = 3 // Extra padding for ' ', '=', and '\n'
	size := len(k) + len(v) + padding
	size += len(strconv.Itoa(size))
	record := fmt.Sprintf("%d %s=%s\n", size, k, v)

	// Final adjustment if adding size field increased the record size.
	if len(record) != size {
		size = len(record)
		record = fmt.Sprintf("%d %s=%s\n", size, k, v)
	}
	return record
}
