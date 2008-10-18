// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

// Count UTF-8 sequences in s.
// Assumes s is well-formed.
export func utflen(s string) int {
	n := 0;
	for i := 0; i < len(s); i++ {
		if s[i]&0xC0 != 0x80 {
			n++
		}
	}
	return n
}

// Split string into array of UTF-8 sequences (still strings)
export func explode(s string) *[]string {
	a := new([]string, utflen(s));
	j := 0;
	for i := 0; i < len(a); i++ {
		ej := j;
		ej++;
		for ej < len(s) && (s[ej]&0xC0) == 0x80 {
			ej++
		}
		a[i] = s[j:ej];
		j = ej
	}
	return a
}

// Count non-overlapping instances of sep in s.
export func count(s, sep string) int {
	if sep == "" {
		return utflen(s)+1
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
export func split(s, sep string) *[]string {
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
export func join(a *[]string, sep string) string {
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

// Convert decimal string to integer.
// TODO: Doesn't check for overflow.
export func atol(s string) (i int64, ok bool) {
	// empty string bad
	if len(s) == 0 { 
		return 0, false
	}
	
	// pick off leading sign
	neg := false;
	if s[0] == '+' {
		s = s[1:len(s)]
	} else if s[0] == '-' {
		neg = true;
		s = s[1:len(s)]
	}
	
	// empty string bad
	if len(s) == 0 { 
		return 0, false
	}

	// pick off zero
	if s == "0" {
		return 0, true
	}
	
	// otherwise, leading zero bad
	if s[0] == '0' {
		return 0, false
	}

	// parse number
	n := int64(0);
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
		n = n*10 + int64(s[i] - '0')
	}
	if neg {
		n = -n
	}
	return n, true
}

export func atoi(s string) (i int, ok bool) {
	ii, okok := atol(s);
	i = int32(ii);
	return i, okok
}

export func itol(i int64) string {
	if i == 0 {
		return "0"
	}
	
	neg := false;	// negative
	u := uint(i);
	if i < 0 {
		neg = true;
		u = -u;
	}

	// Assemble decimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; u > 0; u /= 10 {
		bp--;
		b[bp] = byte(u%10) + '0'
	}
	if neg {	// add sign
		bp--;
		b[bp] = '-'
	}
	
	// BUG return string(b[bp:len(b)])
	return string((&b)[bp:len(b)])
}

export func itoa(i int) string {
	return itol(int64(i));
}
