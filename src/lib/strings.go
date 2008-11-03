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

// Convert decimal string to unsigned integer.
// TODO: Doesn't check for overflow.
export func atoui64(s string) (i uint64, ok bool) {
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
	n := uint64(0);
	for i := 0; i < len(s); i++ {
		if s[i] < '0' || s[i] > '9' {
			return 0, false
		}
		n = n*10 + uint64(s[i] - '0')
	}
	return n, true
}

// Convert decimal string to integer.
// TODO: Doesn't check for overflow.
export func atoi64(s string) (i int64, ok bool) {
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

	var un uint64;
	un, ok = atoui64(s);
	if !ok {
		return 0, false
	}
	n := int64(un);
	if neg {
		n = -n
	}
	return n, true
}

export func atoui(s string) (i uint, ok bool) {
	ii, okok := atoui64(s);
	i = uint(ii);
	return i, okok
}

export func atoi(s string) (i int, ok bool) {
	ii, okok := atoi64(s);
	i = int(ii);
	return i, okok
}

export func ltoa(i int64) string {
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
	return ltoa(int64(i));
}

// Convert float64 to string.  No control over format.
// Result not great; only useful for simple debugging.
export func dtoa(v float64) string {
	var buf [20]byte;

	const n = 7;	// digits printed
	e := 0;	// exp
	var sign byte = '+';
	if(v != 0) {
		// sign
		if(v < 0) {
			v = -v;
			sign = '-';
		}

		// normalize
		for v >= 10 {
			e++;
			v /= 10;
		}
		for v < 1 {
			e--;
			v *= 10;
		}

		// round
		var h float64 = 5;
		for i := 0; i < n; i++ {
			h /= 10;
		}
		v += h;
		if v >= 10 {
			e++;
			v /= 10;
		}
	}

	// format +d.dddd+edd
	buf[0] = sign;
	for i := 0; i < n; i++ {
		s := int64(v);
		buf[i+2] = byte(s)+'0';
		v -= float64(s);
		v *= 10;
	}
	buf[1] = buf[2];
	buf[2] = '.';

	buf[n+2] = 'e';
	buf[n+3] = '+';
	if e < 0 {
		e = -e;
		buf[n+3] = '-';
	}

	// TODO: exponents > 99?
	buf[n+4] = byte((e/10) + '0');
	buf[n+5] = byte((e%10) + '0');
	return string(buf)[0:n+6];	// TODO: should be able to slice buf
}

export func ftoa(v float) string {
	return dtoa(float64(v));
}
