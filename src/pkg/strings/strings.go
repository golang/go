// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package of simple functions to manipulate strings.
package strings

import (
	"unicode";
	"utf8";
)

// explode splits s into an array of UTF-8 sequences, one per Unicode character (still strings) up to a maximum of n (n <= 0 means no limit).
// Invalid UTF-8 sequences become correct encodings of U+FFF8.
func explode(s string, n int) []string {
	if n <= 0 {
		n = len(s);
	}
	a := make([]string, n);
	var size, rune int;
	na := 0;
	for len(s) > 0 {
		if na+1 >= n {
			a[na] = s;
			na++;
			break
		}
		rune, size = utf8.DecodeRuneInString(s);
		s = s[size:len(s)];
		a[na] = string(rune);
		na++;
	}
	return a[0:na]
}

// Count counts the number of non-overlapping instances of sep in s.
func Count(s, sep string) int {
	if sep == "" {
		return utf8.RuneCountInString(s)+1
	}
	c := sep[0];
	n := 0;
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i] == c && (len(sep) == 1 || s[i:i+len(sep)] == sep) {
			n++;
			i += len(sep)-1
		}
	}
	return n
}

// Index returns the index of the first instance of sep in s, or -1 if sep is not present in s.
func Index(s, sep string) int {
	n := len(sep);
	if n == 0 {
		return 0
	}
	c := sep[0];
	for i := 0; i+n <= len(s); i++ {
		if s[i] == c && (n == 1 || s[i:i+n] == sep) {
			return i
		}
	}
	return -1
}

// Index returns the index of the last instance of sep in s, or -1 if sep is not present in s.
func LastIndex(s, sep string) int {
	n := len(sep);
	if n == 0 {
		return len(s)
	}
	c := sep[0];
	for i := len(s)-n; i >= 0; i-- {
		if s[i] == c && (n == 1 || s[i:i+n] == sep) {
			return i
		}
	}
	return -1
}

// Split splits the string s around each instance of sep, returning an array of substrings of s.
// If sep is empty, Split splits s after each UTF-8 sequence.
// If n > 0, split Splits s into at most n substrings; the last subarray will contain an unsplit remainder string.
func Split(s, sep string, n int) []string {
	if sep == "" {
		return explode(s, n)
	}
	if n <= 0 {
		n = Count(s, sep) + 1;
	}
	c := sep[0];
	start := 0;
	a := make([]string, n);
	na := 0;
	for i := 0; i+len(sep) <= len(s) && na+1 < n; i++ {
		if s[i] == c && (len(sep) == 1 || s[i:i+len(sep)] == sep) {
			a[na] = s[start:i];
			na++;
			start = i+len(sep);
			i += len(sep)-1;
		}
	}
	a[na] = s[start:len(s)];
	return a[0:na+1]
}

// Join concatenates the elements of a to create a single string.   The separator string
// sep is placed between elements in the resulting string.
func Join(a []string, sep string) string {
	if len(a) == 0 {
		return ""
	}
	if len(a) == 1 {
		return a[0]
	}
	n := len(sep) * (len(a)-1);
	for i := 0; i < len(a); i++ {
		n += len(a[i])
	}

	b := make([]byte, n);
	bp := 0;
	for i := 0; i < len(a); i++ {
		s := a[i];
		for j := 0; j < len(s); j++ {
			b[bp] = s[j];
			bp++
		}
		if i + 1 < len(a) {
			s = sep;
			for j := 0; j < len(s); j++ {
				b[bp] = s[j];
				bp++
			}
		}
	}
	return string(b)
}

// HasPrefix tests whether the string s begins with prefix.
func HasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[0:len(prefix)] == prefix
}

// HasSuffix tests whether the string s ends with suffix.
func HasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):len(s)] == suffix
}

// Map returns a copy of the string s with all its characters modified
// according to the mapping function.
func Map(mapping func(rune int) int, s string) string {
	// In the worst case, the string can grow when mapped, making
	// things unpleasant.  But it's so rare we barge in assuming it's
	// fine.  It could also shrink but that falls out naturally.
	maxbytes := len(s);	// length of b
	nbytes := 0;	// number of bytes encoded in b
	b := make([]byte, maxbytes);
	for _, c := range s {
		rune := mapping(c);
		wid := 1;
		if rune >= utf8.RuneSelf {
			wid = utf8.RuneLen(rune);
		}
		if nbytes + wid > maxbytes {
			// Grow the buffer.
			maxbytes = maxbytes*2 + utf8.UTFMax;
			nb := make([]byte, maxbytes);
			for i, c := range b[0:nbytes] {
				nb[i] = c
			}
			b = nb;
		}
		nbytes += utf8.EncodeRune(rune, b[nbytes:maxbytes]);
	}
	return string(b[0:nbytes]);
}

// ToUpper returns a copy of the string s with all Unicode letters mapped to their upper case.
func ToUpper(s string) string {
	return Map(unicode.ToUpper, s)
}

// ToUpper returns a copy of the string s with all Unicode letters mapped to their lower case.
func ToLower(s string) string {
	return Map(unicode.ToLower, s)
}

// ToTitle returns a copy of the string s with all Unicode letters mapped to their title case.
func ToTitle(s string) string {
	return Map(unicode.ToTitle, s)
}

// Trim returns a slice of the string s, with all leading and trailing white space
// removed, as defined by Unicode.
func TrimSpace(s string) string {
	start, end := 0, len(s);
	for start < end {
		wid := 1;
		rune := int(s[start]);
		if rune >= utf8.RuneSelf {
			rune, wid = utf8.DecodeRuneInString(s[start:end])
		}
		if !unicode.IsSpace(rune) {
			break;
		}
		start += wid;
	}
	for start < end {
		wid := 1;
		rune := int(s[end-1]);
		if rune >= utf8.RuneSelf {
			// Back up carefully looking for beginning of rune. Mustn't pass start.
			for wid = 2; start <= end-wid && !utf8.RuneStart(s[end-wid]); wid++ {
			}
			if start > end-wid {	// invalid UTF-8 sequence; stop processing
				return s[start:end]
			}
			rune, wid = utf8.DecodeRuneInString(s[end-wid:end]);
		}
		if !unicode.IsSpace(rune) {
			break;
		}
		end -= wid;
	}
	return s[start:end];
}

// Bytes returns a new slice containing the bytes in s.
func Bytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i];
	}
	return b;
}
