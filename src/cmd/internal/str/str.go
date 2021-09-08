// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package str provides string manipulation utilities.
package str

import (
	"bytes"
	"flag"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// StringList flattens its arguments into a single []string.
// Each argument in args must have type string or []string.
func StringList(args ...interface{}) []string {
	var x []string
	for _, arg := range args {
		switch arg := arg.(type) {
		case []string:
			x = append(x, arg...)
		case string:
			x = append(x, arg)
		default:
			panic("stringList: invalid argument of type " + fmt.Sprintf("%T", arg))
		}
	}
	return x
}

// ToFold returns a string with the property that
//	strings.EqualFold(s, t) iff ToFold(s) == ToFold(t)
// This lets us test a large set of strings for fold-equivalent
// duplicates without making a quadratic number of calls
// to EqualFold. Note that strings.ToUpper and strings.ToLower
// do not have the desired property in some corner cases.
func ToFold(s string) string {
	// Fast path: all ASCII, no upper case.
	// Most paths look like this already.
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= utf8.RuneSelf || 'A' <= c && c <= 'Z' {
			goto Slow
		}
	}
	return s

Slow:
	var buf bytes.Buffer
	for _, r := range s {
		// SimpleFold(x) cycles to the next equivalent rune > x
		// or wraps around to smaller values. Iterate until it wraps,
		// and we've found the minimum value.
		for {
			r0 := r
			r = unicode.SimpleFold(r0)
			if r <= r0 {
				break
			}
		}
		// Exception to allow fast path above: A-Z => a-z
		if 'A' <= r && r <= 'Z' {
			r += 'a' - 'A'
		}
		buf.WriteRune(r)
	}
	return buf.String()
}

// FoldDup reports a pair of strings from the list that are
// equal according to strings.EqualFold.
// It returns "", "" if there are no such strings.
func FoldDup(list []string) (string, string) {
	clash := map[string]string{}
	for _, s := range list {
		fold := ToFold(s)
		if t := clash[fold]; t != "" {
			if s > t {
				s, t = t, s
			}
			return s, t
		}
		clash[fold] = s
	}
	return "", ""
}

// Contains reports whether x contains s.
func Contains(x []string, s string) bool {
	for _, t := range x {
		if t == s {
			return true
		}
	}
	return false
}

// Uniq removes consecutive duplicate strings from ss.
func Uniq(ss *[]string) {
	if len(*ss) <= 1 {
		return
	}
	uniq := (*ss)[:1]
	for _, s := range *ss {
		if s != uniq[len(uniq)-1] {
			uniq = append(uniq, s)
		}
	}
	*ss = uniq
}

func isSpaceByte(c byte) bool {
	return c == ' ' || c == '\t' || c == '\n' || c == '\r'
}

// SplitQuotedFields splits s into a list of fields,
// allowing single or double quotes around elements.
// There is no unescaping or other processing within
// quoted fields.
func SplitQuotedFields(s string) ([]string, error) {
	// Split fields allowing '' or "" around elements.
	// Quotes further inside the string do not count.
	var f []string
	for len(s) > 0 {
		for len(s) > 0 && isSpaceByte(s[0]) {
			s = s[1:]
		}
		if len(s) == 0 {
			break
		}
		// Accepted quoted string. No unescaping inside.
		if s[0] == '"' || s[0] == '\'' {
			quote := s[0]
			s = s[1:]
			i := 0
			for i < len(s) && s[i] != quote {
				i++
			}
			if i >= len(s) {
				return nil, fmt.Errorf("unterminated %c string", quote)
			}
			f = append(f, s[:i])
			s = s[i+1:]
			continue
		}
		i := 0
		for i < len(s) && !isSpaceByte(s[i]) {
			i++
		}
		f = append(f, s[:i])
		s = s[i:]
	}
	return f, nil
}

// JoinAndQuoteFields joins a list of arguments into a string that can be parsed
// with SplitQuotedFields. Arguments are quoted only if necessary; arguments
// without spaces or quotes are kept as-is. No argument may contain both
// single and double quotes.
func JoinAndQuoteFields(args []string) (string, error) {
	var buf []byte
	for i, arg := range args {
		if i > 0 {
			buf = append(buf, ' ')
		}
		var sawSpace, sawSingleQuote, sawDoubleQuote bool
		for _, c := range arg {
			switch {
			case c > unicode.MaxASCII:
				continue
			case isSpaceByte(byte(c)):
				sawSpace = true
			case c == '\'':
				sawSingleQuote = true
			case c == '"':
				sawDoubleQuote = true
			}
		}
		switch {
		case !sawSpace && !sawSingleQuote && !sawDoubleQuote:
			buf = append(buf, []byte(arg)...)

		case !sawSingleQuote:
			buf = append(buf, '\'')
			buf = append(buf, []byte(arg)...)
			buf = append(buf, '\'')

		case !sawDoubleQuote:
			buf = append(buf, '"')
			buf = append(buf, []byte(arg)...)
			buf = append(buf, '"')

		default:
			return "", fmt.Errorf("argument %q contains both single and double quotes and cannot be quoted", arg)
		}
	}
	return string(buf), nil
}

// A QuotedStringListFlag parses a list of string arguments encoded with
// JoinAndQuoteFields. It is useful for flags like cmd/link's -extldflags.
type QuotedStringListFlag []string

var _ flag.Value = (*QuotedStringListFlag)(nil)

func (f *QuotedStringListFlag) Set(v string) error {
	fs, err := SplitQuotedFields(v)
	if err != nil {
		return err
	}
	*f = fs[:len(fs):len(fs)]
	return nil
}

func (f *QuotedStringListFlag) String() string {
	if f == nil {
		return ""
	}
	s, err := JoinAndQuoteFields(*f)
	if err != nil {
		return strings.Join(*f, " ")
	}
	return s
}
