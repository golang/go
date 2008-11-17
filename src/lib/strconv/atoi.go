// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

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
