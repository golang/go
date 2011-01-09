// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The bytes package implements functions for the manipulation of byte slices.
// Analogous to the facilities of the strings package.
package bytes

import (
	"unicode"
	"utf8"
)

// Compare returns an integer comparing the two byte arrays lexicographically.
// The result will be 0 if a==b, -1 if a < b, and +1 if a > b
func Compare(a, b []byte) int {
	m := len(a)
	if m > len(b) {
		m = len(b)
	}
	for i, ac := range a[0:m] {
		bc := b[i]
		switch {
		case ac > bc:
			return 1
		case ac < bc:
			return -1
		}
	}
	switch {
	case len(a) < len(b):
		return -1
	case len(a) > len(b):
		return 1
	}
	return 0
}

// Equal returns a boolean reporting whether a == b.
func Equal(a, b []byte) bool {
	if len(a) != len(b) {
		return false
	}
	for i, c := range a {
		if c != b[i] {
			return false
		}
	}
	return true
}

// explode splits s into an array of UTF-8 sequences, one per Unicode character (still arrays of bytes),
// up to a maximum of n byte arrays. Invalid UTF-8 sequences are chopped into individual bytes.
func explode(s []byte, n int) [][]byte {
	if n <= 0 {
		n = len(s)
	}
	a := make([][]byte, n)
	var size int
	na := 0
	for len(s) > 0 {
		if na+1 >= n {
			a[na] = s
			na++
			break
		}
		_, size = utf8.DecodeRune(s)
		a[na] = s[0:size]
		s = s[size:]
		na++
	}
	return a[0:na]
}

// Count counts the number of non-overlapping instances of sep in s.
func Count(s, sep []byte) int {
	if len(sep) == 0 {
		return utf8.RuneCount(s) + 1
	}
	c := sep[0]
	n := 0
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i] == c && (len(sep) == 1 || Equal(s[i:i+len(sep)], sep)) {
			n++
			i += len(sep) - 1
		}
	}
	return n
}

// Index returns the index of the first instance of sep in s, or -1 if sep is not present in s.
func Index(s, sep []byte) int {
	n := len(sep)
	if n == 0 {
		return 0
	}
	c := sep[0]
	for i := 0; i+n <= len(s); i++ {
		if s[i] == c && (n == 1 || Equal(s[i:i+n], sep)) {
			return i
		}
	}
	return -1
}

func indexBytePortable(s []byte, c byte) int {
	for i, b := range s {
		if b == c {
			return i
		}
	}
	return -1
}

// LastIndex returns the index of the last instance of sep in s, or -1 if sep is not present in s.
func LastIndex(s, sep []byte) int {
	n := len(sep)
	if n == 0 {
		return len(s)
	}
	c := sep[0]
	for i := len(s) - n; i >= 0; i-- {
		if s[i] == c && (n == 1 || Equal(s[i:i+n], sep)) {
			return i
		}
	}
	return -1
}

// IndexRune interprets s as a sequence of UTF-8-encoded Unicode code points.
// It returns the byte index of the first occurrence in s of the given rune.
// It returns -1 if rune is not present in s.
func IndexRune(s []byte, rune int) int {
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRune(s[i:])
		if r == rune {
			return i
		}
		i += size
	}
	return -1
}

// IndexAny interprets s as a sequence of UTF-8-encoded Unicode code points.
// It returns the byte index of the first occurrence in s of any of the Unicode
// code points in chars.  It returns -1 if chars is empty or if there is no code
// point in common.
func IndexAny(s []byte, chars string) int {
	if len(chars) > 0 {
		var rune, width int
		for i := 0; i < len(s); i += width {
			rune = int(s[i])
			if rune < utf8.RuneSelf {
				width = 1
			} else {
				rune, width = utf8.DecodeRune(s[i:])
			}
			for _, r := range chars {
				if rune == r {
					return i
				}
			}
		}
	}
	return -1
}

// LastIndexAny interprets s as a sequence of UTF-8-encoded Unicode code
// points.  It returns the byte index of the last occurrence in s of any of
// the Unicode code points in chars.  It returns -1 if chars is empty or if
// there is no code point in common.
func LastIndexAny(s []byte, chars string) int {
	if len(chars) > 0 {
		for i := len(s); i > 0; {
			rune, size := utf8.DecodeLastRune(s[0:i])
			i -= size
			for _, m := range chars {
				if rune == m {
					return i
				}
			}
		}
	}
	return -1
}

// Generic split: splits after each instance of sep,
// including sepSave bytes of sep in the subarrays.
func genSplit(s, sep []byte, sepSave, n int) [][]byte {
	if n == 0 {
		return nil
	}
	if len(sep) == 0 {
		return explode(s, n)
	}
	if n < 0 {
		n = Count(s, sep) + 1
	}
	c := sep[0]
	start := 0
	a := make([][]byte, n)
	na := 0
	for i := 0; i+len(sep) <= len(s) && na+1 < n; i++ {
		if s[i] == c && (len(sep) == 1 || Equal(s[i:i+len(sep)], sep)) {
			a[na] = s[start : i+sepSave]
			na++
			start = i + len(sep)
			i += len(sep) - 1
		}
	}
	a[na] = s[start:]
	return a[0 : na+1]
}

// Split slices s into subslices separated by sep and returns a slice of
// the subslices between those separators.
// If sep is empty, Split splits after each UTF-8 sequence.
// The count determines the number of subslices to return:
//   n > 0: at most n subslices; the last subslice will be the unsplit remainder.
//   n == 0: the result is nil (zero subslices)
//   n < 0: all subslices
func Split(s, sep []byte, n int) [][]byte { return genSplit(s, sep, 0, n) }

// SplitAfter slices s into subslices after each instance of sep and
// returns a slice of those subslices.
// If sep is empty, Split splits after each UTF-8 sequence.
// The count determines the number of subslices to return:
//   n > 0: at most n subslices; the last subslice will be the unsplit remainder.
//   n == 0: the result is nil (zero subslices)
//   n < 0: all subslices
func SplitAfter(s, sep []byte, n int) [][]byte {
	return genSplit(s, sep, len(sep), n)
}

// Fields splits the array s around each instance of one or more consecutive white space
// characters, returning a slice of subarrays of s or an empty list if s contains only white space.
func Fields(s []byte) [][]byte {
	return FieldsFunc(s, unicode.IsSpace)
}

// FieldsFunc interprets s as a sequence of UTF-8-encoded Unicode code points.
// It splits the array s at each run of code points c satisfying f(c) and
// returns a slice of subarrays of s.  If no code points in s satisfy f(c), an
// empty slice is returned.
func FieldsFunc(s []byte, f func(int) bool) [][]byte {
	n := 0
	inField := false
	for i := 0; i < len(s); {
		rune, size := utf8.DecodeRune(s[i:])
		wasInField := inField
		inField = !f(rune)
		if inField && !wasInField {
			n++
		}
		i += size
	}

	a := make([][]byte, n)
	na := 0
	fieldStart := -1
	for i := 0; i <= len(s) && na < n; {
		rune, size := utf8.DecodeRune(s[i:])
		if fieldStart < 0 && size > 0 && !f(rune) {
			fieldStart = i
			i += size
			continue
		}
		if fieldStart >= 0 && (size == 0 || f(rune)) {
			a[na] = s[fieldStart:i]
			na++
			fieldStart = -1
		}
		if size == 0 {
			break
		}
		i += size
	}
	return a[0:na]
}

// Join concatenates the elements of a to create a single byte array.   The separator
// sep is placed between elements in the resulting array.
func Join(a [][]byte, sep []byte) []byte {
	if len(a) == 0 {
		return []byte{}
	}
	if len(a) == 1 {
		return a[0]
	}
	n := len(sep) * (len(a) - 1)
	for i := 0; i < len(a); i++ {
		n += len(a[i])
	}

	b := make([]byte, n)
	bp := 0
	for i := 0; i < len(a); i++ {
		s := a[i]
		for j := 0; j < len(s); j++ {
			b[bp] = s[j]
			bp++
		}
		if i+1 < len(a) {
			s = sep
			for j := 0; j < len(s); j++ {
				b[bp] = s[j]
				bp++
			}
		}
	}
	return b
}

// HasPrefix tests whether the byte array s begins with prefix.
func HasPrefix(s, prefix []byte) bool {
	return len(s) >= len(prefix) && Equal(s[0:len(prefix)], prefix)
}

// HasSuffix tests whether the byte array s ends with suffix.
func HasSuffix(s, suffix []byte) bool {
	return len(s) >= len(suffix) && Equal(s[len(s)-len(suffix):], suffix)
}

// Map returns a copy of the byte array s with all its characters modified
// according to the mapping function. If mapping returns a negative value, the character is
// dropped from the string with no replacement.  The characters in s and the
// output are interpreted as UTF-8-encoded Unicode code points.
func Map(mapping func(rune int) int, s []byte) []byte {
	// In the worst case, the array can grow when mapped, making
	// things unpleasant.  But it's so rare we barge in assuming it's
	// fine.  It could also shrink but that falls out naturally.
	maxbytes := len(s) // length of b
	nbytes := 0        // number of bytes encoded in b
	b := make([]byte, maxbytes)
	for i := 0; i < len(s); {
		wid := 1
		rune := int(s[i])
		if rune >= utf8.RuneSelf {
			rune, wid = utf8.DecodeRune(s[i:])
		}
		rune = mapping(rune)
		if rune >= 0 {
			if nbytes+utf8.RuneLen(rune) > maxbytes {
				// Grow the buffer.
				maxbytes = maxbytes*2 + utf8.UTFMax
				nb := make([]byte, maxbytes)
				copy(nb, b[0:nbytes])
				b = nb
			}
			nbytes += utf8.EncodeRune(b[nbytes:maxbytes], rune)
		}
		i += wid
	}
	return b[0:nbytes]
}

// Repeat returns a new byte slice consisting of count copies of b.
func Repeat(b []byte, count int) []byte {
	nb := make([]byte, len(b)*count)
	bp := 0
	for i := 0; i < count; i++ {
		for j := 0; j < len(b); j++ {
			nb[bp] = b[j]
			bp++
		}
	}
	return nb
}

// ToUpper returns a copy of the byte array s with all Unicode letters mapped to their upper case.
func ToUpper(s []byte) []byte { return Map(unicode.ToUpper, s) }

// ToUpper returns a copy of the byte array s with all Unicode letters mapped to their lower case.
func ToLower(s []byte) []byte { return Map(unicode.ToLower, s) }

// ToTitle returns a copy of the byte array s with all Unicode letters mapped to their title case.
func ToTitle(s []byte) []byte { return Map(unicode.ToTitle, s) }

// ToUpperSpecial returns a copy of the byte array s with all Unicode letters mapped to their
// upper case, giving priority to the special casing rules.
func ToUpperSpecial(_case unicode.SpecialCase, s []byte) []byte {
	return Map(func(r int) int { return _case.ToUpper(r) }, s)
}

// ToLowerSpecial returns a copy of the byte array s with all Unicode letters mapped to their
// lower case, giving priority to the special casing rules.
func ToLowerSpecial(_case unicode.SpecialCase, s []byte) []byte {
	return Map(func(r int) int { return _case.ToLower(r) }, s)
}

// ToTitleSpecial returns a copy of the byte array s with all Unicode letters mapped to their
// title case, giving priority to the special casing rules.
func ToTitleSpecial(_case unicode.SpecialCase, s []byte) []byte {
	return Map(func(r int) int { return _case.ToTitle(r) }, s)
}


// isSeparator reports whether the rune could mark a word boundary.
// TODO: update when package unicode captures more of the properties.
func isSeparator(rune int) bool {
	// ASCII alphanumerics and underscore are not separators
	if rune <= 0x7F {
		switch {
		case '0' <= rune && rune <= '9':
			return false
		case 'a' <= rune && rune <= 'z':
			return false
		case 'A' <= rune && rune <= 'Z':
			return false
		case rune == '_':
			return false
		}
		return true
	}
	// Letters and digits are not separators
	if unicode.IsLetter(rune) || unicode.IsDigit(rune) {
		return false
	}
	// Otherwise, all we can do for now is treat spaces as separators.
	return unicode.IsSpace(rune)
}

// BUG(r): The rule Title uses for word boundaries does not handle Unicode punctuation properly.

// Title returns a copy of s with all Unicode letters that begin words
// mapped to their title case.
func Title(s []byte) []byte {
	// Use a closure here to remember state.
	// Hackish but effective. Depends on Map scanning in order and calling
	// the closure once per rune.
	prev := ' '
	return Map(
		func(r int) int {
			if isSeparator(prev) {
				prev = r
				return unicode.ToTitle(r)
			}
			prev = r
			return r
		},
		s)
}

// TrimLeftFunc returns a subslice of s by slicing off all leading UTF-8-encoded
// Unicode code points c that satisfy f(c).
func TrimLeftFunc(s []byte, f func(r int) bool) []byte {
	i := indexFunc(s, f, false)
	if i == -1 {
		return nil
	}
	return s[i:]
}

// TrimRightFunc returns a subslice of s by slicing off all trailing UTF-8
// encoded Unicode code points c that satisfy f(c).
func TrimRightFunc(s []byte, f func(r int) bool) []byte {
	i := lastIndexFunc(s, f, false)
	if i >= 0 && s[i] >= utf8.RuneSelf {
		_, wid := utf8.DecodeRune(s[i:])
		i += wid
	} else {
		i++
	}
	return s[0:i]
}

// TrimFunc returns a subslice of s by slicing off all leading and trailing
// UTF-8-encoded Unicode code points c that satisfy f(c).
func TrimFunc(s []byte, f func(r int) bool) []byte {
	return TrimRightFunc(TrimLeftFunc(s, f), f)
}

// IndexFunc interprets s as a sequence of UTF-8-encoded Unicode code points.
// It returns the byte index in s of the first Unicode
// code point satisfying f(c), or -1 if none do.
func IndexFunc(s []byte, f func(r int) bool) int {
	return indexFunc(s, f, true)
}

// LastIndexFunc interprets s as a sequence of UTF-8-encoded Unicode code points.
// It returns the byte index in s of the last Unicode
// code point satisfying f(c), or -1 if none do.
func LastIndexFunc(s []byte, f func(r int) bool) int {
	return lastIndexFunc(s, f, true)
}

// indexFunc is the same as IndexFunc except that if
// truth==false, the sense of the predicate function is
// inverted.
func indexFunc(s []byte, f func(r int) bool, truth bool) int {
	start := 0
	for start < len(s) {
		wid := 1
		rune := int(s[start])
		if rune >= utf8.RuneSelf {
			rune, wid = utf8.DecodeRune(s[start:])
		}
		if f(rune) == truth {
			return start
		}
		start += wid
	}
	return -1
}

// lastIndexFunc is the same as LastIndexFunc except that if
// truth==false, the sense of the predicate function is
// inverted.
func lastIndexFunc(s []byte, f func(r int) bool, truth bool) int {
	for i := len(s); i > 0; {
		rune, size := utf8.DecodeLastRune(s[0:i])
		i -= size
		if f(rune) == truth {
			return i
		}
	}
	return -1
}

func makeCutsetFunc(cutset string) func(rune int) bool {
	return func(rune int) bool {
		for _, c := range cutset {
			if c == rune {
				return true
			}
		}
		return false
	}
}

// Trim returns a subslice of s by slicing off all leading and
// trailing UTF-8-encoded Unicode code points contained in cutset.
func Trim(s []byte, cutset string) []byte {
	return TrimFunc(s, makeCutsetFunc(cutset))
}

// TrimLeft returns a subslice of s by slicing off all leading
// UTF-8-encoded Unicode code points contained in cutset.
func TrimLeft(s []byte, cutset string) []byte {
	return TrimLeftFunc(s, makeCutsetFunc(cutset))
}

// TrimRight returns a subslice of s by slicing off all trailing
// UTF-8-encoded Unicode code points that are contained in cutset.
func TrimRight(s []byte, cutset string) []byte {
	return TrimRightFunc(s, makeCutsetFunc(cutset))
}

// TrimSpace returns a subslice of s by slicing off all leading and
// trailing white space, as defined by Unicode.
func TrimSpace(s []byte) []byte {
	return TrimFunc(s, unicode.IsSpace)
}

// Runes returns a slice of runes (Unicode code points) equivalent to s.
func Runes(s []byte) []int {
	t := make([]int, utf8.RuneCount(s))
	i := 0
	for len(s) > 0 {
		r, l := utf8.DecodeRune(s)
		t[i] = r
		i++
		s = s[l:]
	}
	return t
}

// Replace returns a copy of the slice s with the first n
// non-overlapping instances of old replaced by new.
// If n < 0, there is no limit on the number of replacements.
func Replace(s, old, new []byte, n int) []byte {
	if n == 0 {
		return s // avoid allocation
	}
	// Compute number of replacements.
	if m := Count(s, old); m == 0 {
		return s // avoid allocation
	} else if n <= 0 || m < n {
		n = m
	}

	// Apply replacements to buffer.
	t := make([]byte, len(s)+n*(len(new)-len(old)))
	w := 0
	start := 0
	for i := 0; i < n; i++ {
		j := start
		if len(old) == 0 {
			if i > 0 {
				_, wid := utf8.DecodeRune(s[start:])
				j += wid
			}
		} else {
			j += Index(s[start:], old)
		}
		w += copy(t[w:], s[start:j])
		w += copy(t[w:], new)
		start = j + len(old)
	}
	w += copy(t[w:], s[start:])
	return t[0:w]
}
