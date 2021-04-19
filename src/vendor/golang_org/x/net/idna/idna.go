// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package idna implements IDNA2008 (Internationalized Domain Names for
// Applications), defined in RFC 5890, RFC 5891, RFC 5892, RFC 5893 and
// RFC 5894.
package idna // import "golang.org/x/net/idna"

import (
	"strings"
	"unicode/utf8"
)

// TODO(nigeltao): specify when errors occur. For example, is ToASCII(".") or
// ToASCII("foo\x00") an error? See also http://www.unicode.org/faq/idn.html#11

// acePrefix is the ASCII Compatible Encoding prefix.
const acePrefix = "xn--"

// ToASCII converts a domain or domain label to its ASCII form. For example,
// ToASCII("bücher.example.com") is "xn--bcher-kva.example.com", and
// ToASCII("golang") is "golang".
func ToASCII(s string) (string, error) {
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

// ToUnicode converts a domain or domain label to its Unicode form. For example,
// ToUnicode("xn--bcher-kva.example.com") is "bücher.example.com", and
// ToUnicode("golang") is "golang".
func ToUnicode(s string) (string, error) {
	if !strings.Contains(s, acePrefix) {
		return s, nil
	}
	labels := strings.Split(s, ".")
	for i, label := range labels {
		if strings.HasPrefix(label, acePrefix) {
			u, err := decode(label[len(acePrefix):])
			if err != nil {
				return "", err
			}
			labels[i] = u
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
