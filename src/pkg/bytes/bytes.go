// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package of simple functions to manipulate arrays of bytes.
// Analagous to the facilities of the strings package.
package bytes

import "utf8"

// Compare returns an integer comparing the two byte arrays lexicographically.
// The result will be 0 if a==b, -1 if a < b, and +1 if a > b
func Compare(a, b []byte) int {
	for i := 0; i < len(a) && i < len(b); i++ {
		switch {
		case a[i] > b[i]:
			return 1
		case a[i] < b[i]:
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
	for i := 0; i < len(a); i++ {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// Copy copies bytes from src to dst,
// stopping when either all of src has been copied
// or all of dst has been filled.
// It returns the number of bytes copied.
func Copy(dst, src []byte) int {
	if len(src) > len(dst) {
		src = src[0:len(dst)];
	}
	for i, x := range src {
		dst[i] = x
	}
	return len(src)
}

// explode splits s into an array of UTF-8 sequences, one per Unicode character (still arrays of bytes),
// up to a maximum of n byte arrays. Invalid UTF-8 sequences are chopped into individual bytes.
func explode(s []byte, n int) [][]byte {
	if n <= 0 {
		n = len(s);
	}
	a := make([][]byte, n);
	var size, rune int;
	na := 0;
	for len(s) > 0 {
		if na+1 >= n {
			a[na] = s;
			na++;
			break
		}
		rune, size = utf8.DecodeRune(s);
		a[na] = s[0:size];
		s = s[size:len(s)];
		na++;
	}
	return a[0:na]
}

// Count counts the number of non-overlapping instances of sep in s.
func Count(s, sep []byte) int {
	if len(sep) == 0 {
		return utf8.RuneCount(s)+1
	}
	c := sep[0];
	n := 0;
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i] == c && (len(sep) == 1 || Equal(s[i:i+len(sep)], sep)) {
			n++;
			i += len(sep)-1
		}
	}
	return n
}

// Index returns the index of the first instance of sep in s, or -1 if sep is not present in s.
func Index(s, sep []byte) int {
	n := len(sep);
	if n == 0 {
		return 0
	}
	c := sep[0];
	for i := 0; i+n <= len(s); i++ {
		if s[i] == c && (n == 1 || Equal(s[i:i+n], sep)) {
			return i
		}
	}
	return -1
}

// Split splits the array s around each instance of sep, returning an array of subarrays of s.
// If sep is empty, Split splits s after each UTF-8 sequence.
// If n > 0, split Splits s into at most n subarrays; the last subarray will contain an unsplit remainder.
func Split(s, sep []byte, n int) [][]byte {
	if len(sep) == 0 {
		return explode(s, n)
	}
	if n <= 0 {
		n = Count(s, sep) + 1;
	}
	c := sep[0];
	start := 0;
	a := make([][]byte, n);
	na := 0;
	for i := 0; i+len(sep) <= len(s) && na+1 < n; i++ {
		if s[i] == c && (len(sep) == 1 || Equal(s[i:i+len(sep)], sep)) {
			a[na] = s[start:i];
			na++;
			start = i+len(sep);
			i += len(sep)-1;
		}
	}
	a[na] = s[start:len(s)];
	return a[0:na+1]
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
	return b
}

// HasPrefix tests whether the byte array s begins with prefix.
func HasPrefix(s, prefix []byte) bool {
	return len(s) >= len(prefix) && Equal(s[0:len(prefix)], prefix)
}

// HasSuffix tests whether the byte array s ends with suffix.
func HasSuffix(s, suffix []byte) bool {
	return len(s) >= len(suffix) && Equal(s[len(s)-len(suffix):len(s)], suffix)
}
