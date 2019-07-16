// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cookiejar

// This file implements the Punycode algorithm from RFC 3492.

import (
	"fmt"
	"strings"
	"unicode/utf8"
)

// These parameter values are specified in section 5.
//
// All computation is done with int32s, so that overflow behavior is identical
// regardless of whether int is 32-bit or 64-bit.
const (
	base        int32 = 36
	damp        int32 = 700
	initialBias int32 = 72
	initialN    int32 = 128
	skew        int32 = 38
	tmax        int32 = 26
	tmin        int32 = 1
)

// encode encodes a string as specified in section 6.3 and prepends prefix to
// the result.
//
// The "while h < length(input)" line in the specification becomes "for
// remaining != 0" in the Go code, because len(s) in Go is in bytes, not runes.
func encode(prefix, s string) (string, error) {
	output := make([]byte, len(prefix), len(prefix)+1+2*len(s))
	copy(output, prefix)
	delta, n, bias := int32(0), initialN, initialBias
	b, remaining := int32(0), int32(0)
	for _, r := range s {
		if r < utf8.RuneSelf {
			b++
			output = append(output, byte(r))
		} else {
			remaining++
		}
	}
	h := b
	if b > 0 {
		output = append(output, '-')
	}
	for remaining != 0 {
		m := int32(0x7fffffff)
		for _, r := range s {
			if m > r && r >= n {
				m = r
			}
		}
		delta += (m - n) * (h + 1)
		if delta < 0 {
			return "", fmt.Errorf("cookiejar: invalid label %q", s)
		}
		n = m
		for _, r := range s {
			if r < n {
				delta++
				if delta < 0 {
					return "", fmt.Errorf("cookiejar: invalid label %q", s)
				}
				continue
			}
			if r > n {
				continue
			}
			q := delta
			for k := base; ; k += base {
				t := k - bias
				if t < tmin {
					t = tmin
				} else if t > tmax {
					t = tmax
				}
				if q < t {
					break
				}
				output = append(output, encodeDigit(t+(q-t)%(base-t)))
				q = (q - t) / (base - t)
			}
			output = append(output, encodeDigit(q))
			bias = adapt(delta, h+1, h == b)
			delta = 0
			h++
			remaining--
		}
		delta++
		n++
	}
	return string(output), nil
}

func encodeDigit(digit int32) byte {
	switch {
	case 0 <= digit && digit < 26:
		return byte(digit + 'a')
	case 26 <= digit && digit < 36:
		return byte(digit + ('0' - 26))
	}
	panic("cookiejar: internal error in punycode encoding")
}

// adapt is the bias adaptation function specified in section 6.1.
func adapt(delta, numPoints int32, firstTime bool) int32 {
	if firstTime {
		delta /= damp
	} else {
		delta /= 2
	}
	delta += delta / numPoints
	k := int32(0)
	for delta > ((base-tmin)*tmax)/2 {
		delta /= base - tmin
		k += base
	}
	return k + (base-tmin+1)*delta/(delta+skew)
}

// Strictly speaking, the remaining code below deals with IDNA (RFC 5890 and
// friends) and not Punycode (RFC 3492) per se.

// acePrefix is the ASCII Compatible Encoding prefix.
const acePrefix = "xn--"

// toASCII converts a domain or domain label to its ASCII form. For example,
// toASCII("bücher.example.com") is "xn--bcher-kva.example.com", and
// toASCII("golang") is "golang".
func toASCII(s string) (string, error) {
	if ascii(s) {
		return s, nil
	}
	labels := strings.Split(s, ".")
	for i, label := range labels {
		if !ascii(label) {
			a, err := encode(acePrefix, label)
			if err != nil {
				return "", err
			}
			labels[i] = a
		}
	}
	return strings.Join(labels, "."), nil
}

func ascii(s string) bool {
	for i := 0; i < len(s); i++ {
		if s[i] >= utf8.RuneSelf {
			return false
		}
	}
	return true
}
