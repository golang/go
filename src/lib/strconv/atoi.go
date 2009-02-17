// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import "os"

func computeIntsize() uint {
	siz := uint(8);
	for 1<<siz != 0 {
		siz *= 2
	}
	return siz
}
var intsize = computeIntsize();

// Return the first number n such that n*base >= 1<<64.
func cutoff64(base int) uint64 {
	if base < 2 {
		return 0;
	}
	return (1<<64 - 1) / uint64(base) + 1;
}

// Convert arbitrary base string to unsigned integer.
func Btoui64(base int, s string) (n uint64, err *os.Error) {
	if base < 2 || base > 36 || len(s) < 1 {
		return 0, os.EINVAL;
	}

	n = 0;
	cutoff := cutoff64(base);

	for i := 0; i < len(s); i++ {
		var v byte;
		switch {
		case '0' <= s[i] && s[i] <= '9':
			v = s[i] - '0';
		case 'a' <= s[i] && s[i] <= 'z':
			v = s[i] - 'a' + 10;
		case 'A' <= s[i] && s[i] <= 'Z':
			v = s[i] - 'A' + 10;
		default:
			return 0, os.EINVAL;
		}
		if int(v) >= base {
			return 0, os.EINVAL;
		}

		if n >= cutoff {
			// n*base overflows
			return 1<<64-1, os.ERANGE;
		}
		n *= uint64(base);

		n1 := n+uint64(v);
		if n1 < n {
			// n+v overflows
			return 1<<64-1, os.ERANGE;
		}
		n = n1;
	}

	return n, nil;
}


// Convert string to uint64.
// Use standard prefixes to signal octal, hexadecimal.
func Atoui64(s string) (i uint64, err *os.Error) {
	// Empty string bad.
	if len(s) == 0 {
		return 0, os.EINVAL
	}

	// Look for octal, hex prefix.
	if s[0] == '0' && len(s) > 1 {
		if s[1] == 'x' || s[1] == 'X' {
			// hex
			return Btoui64(16, s[2:len(s)]);
		}
		// octal
		return Btoui64(8, s[1:len(s)]);
	}
	// decimal
	return Btoui64(10, s);
}

// Convert string to int64.
// Use standard prefixes to signal octal, hexadecimal.
func Atoi64(s string) (i int64, err *os.Error) {
	// Empty string bad.
	if len(s) == 0 {
		return 0, os.EINVAL
	}

	// Pick off leading sign.
	neg := false;
	if s[0] == '+' {
		s = s[1:len(s)]
	} else if s[0] == '-' {
		neg = true;
		s = s[1:len(s)]
	}

	// Convert unsigned and check range.
	var un uint64;
	un, err = Atoui64(s);
	if err != nil && err != os.ERANGE {
		return 0, err
	}
	if !neg && un >= 1<<63 {
		return 1<<63-1, os.ERANGE
	}
	if neg && un > 1<<63 {
		return -1<<63, os.ERANGE
	}
	n := int64(un);
	if neg {
		n = -n
	}
	return n, nil
}

// Convert string to uint.
// Use standard prefixes to signal octal, hexadecimal.
func Atoui(s string) (i uint, err *os.Error) {
	i1, e1 := Atoui64(s);
	if e1 != nil && e1 != os.ERANGE {
		return 0, e1
	}
	i = uint(i1);
	if uint64(i) != i1 {
		// TODO: return uint(^0), os.ERANGE.
		i1 = 1<<64-1;
		return uint(i1), os.ERANGE
	}
	return i, nil
}

// Convert string to int.
// Use standard prefixes to signal octal, hexadecimal.
func Atoi(s string) (i int, err *os.Error) {
	i1, e1 := Atoi64(s);
	if e1 != nil && e1 != os.ERANGE {
		return 0, e1
	}
	i = int(i1);
	if int64(i) != i1 {
		if i1 < 0 {
			return -1<<(intsize-1), os.ERANGE
		}
		return 1<<(intsize-1) - 1, os.ERANGE
	}
	return i, nil
}

