// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "utf8"

// Split string into array of UTF-8 sequences (still strings)
export func explode(s string) []string {
	a := new([]string, utf8.RuneCountInString(s, 0, len(s)));
	j := 0;
	var size, rune int;
	for i := 0; i < len(a); i++ {
		rune, size = utf8.DecodeRuneInString(s, j);
		a[i] = string(rune);
		j += size;
	}
	return a
}

// Count non-overlapping instances of sep in s.
export func count(s, sep string) int {
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

// Return index of first instance of sep in s.
export func index(s, sep string) int {
	if sep == "" {
		return 0
	}
	c := sep[0];
	for i := 0; i+len(sep) <= len(s); i++ {
		if s[i] == c && (len(sep) == 1 || s[i:i+len(sep)] == sep) {
			return i
		}
	}
	return -1
}

// Split string into list of strings at separators
export func split(s, sep string) []string {
	if sep == "" {
		return explode(s)
	}
	c := sep[0];
	start := 0;
	n := count(s, sep)+1;
	a := new([]string, n);
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

// Join list of strings with separators between them.
export func join(a []string, sep string) string {
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

	b := new([]byte, n);
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
