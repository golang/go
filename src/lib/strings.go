// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package of simple functions to manipulate strings.
package strings

import "utf8"

// Explode splits s into an array of UTF-8 sequences, one per Unicode character (still strings).
// Invalid UTF-8 sequences become correct encodings of U+FFF8.
func Explode(s string) []string {
	a := make([]string, utf8.RuneCountInString(s, 0, len(s)));
	j := 0;
	var size, rune int;
	for i := 0; i < len(a); i++ {
		rune, size = utf8.DecodeRuneInString(s, j);
		a[i] = string(rune);
		j += size;
	}
	return a
}

// Count counts the number of non-overlapping instances of sep in s.
func Count(s, sep string) int {
	if sep == "" {
		return utf8.RuneCountInString(s, 0, len(s))+1
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

// Split returns the array representing the substrings of s separated by string sep. Adjacent
// occurrences of sep produce empty substrings.  If sep is empty, it is the same as Explode.
func Split(s, sep string) []string {
	if sep == "" {
		return Explode(s)
	}
	c := sep[0];
	start := 0;
	n := Count(s, sep)+1;
	a := make([]string, n);
	na := 0;
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i] == c && (len(sep) == 1 || s[i:i+len(sep)] == sep) {
			a[na] = s[start:i];
			na++;
			start = i+len(sep);
			i += len(sep)-1
		}
	}
	a[na] = s[start:len(s)];
	return a
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
