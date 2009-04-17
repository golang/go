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

// Btoui64 interprets a string s in an arbitrary base b (2 to 36)
// and returns the corresponding value n.
//
// Btoui64 returns err == os.EINVAL if b is out of
// range or s is empty or contains invalid digits.
// It returns err == os.ERANGE if the value corresponding
// to s cannot be represented by a uint64.
func Btoui64(s string, b int) (n uint64, err os.Error) {
	if b < 2 || b > 36 || len(s) < 1 {
		return 0, os.EINVAL;
	}

	n = 0;
	cutoff := cutoff64(b);

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
		if int(v) >= b {
			return 0, os.EINVAL;
		}

		if n >= cutoff {
			// n*b overflows
			return 1<<64-1, os.ERANGE;
		}
		n *= uint64(b);

		n1 := n+uint64(v);
		if n1 < n {
			// n+v overflows
			return 1<<64-1, os.ERANGE;
		}
		n = n1;
	}

	return n, nil;
}

// Atoui64 interprets a string s as an unsigned decimal, octal, or
// hexadecimal number and returns the corresponding value n.
// The default base is decimal.  Strings beginning with 0x are
// hexadecimal; strings beginning with 0 are octal.
//
// Atoui64 returns err == os.EINVAL if s is empty or contains invalid digits.
// It returns err == os.ERANGE if s cannot be represented by a uint64.
func Atoui64(s string) (n uint64, err os.Error) {
	// Empty string bad.
	if len(s) == 0 {
		return 0, os.EINVAL
	}

	// Look for octal, hex prefix.
	if s[0] == '0' && len(s) > 1 {
		if s[1] == 'x' || s[1] == 'X' {
			// hex
			return Btoui64(s[2:len(s)], 16);
		}
		// octal
		return Btoui64(s[1:len(s)], 8);
	}
	// decimal
	return Btoui64(s, 10);
}


// Atoi64 is like Atoui64 but allows signed numbers and
// returns its result in an int64.
func Atoi64(s string) (i int64, err os.Error) {
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

// Atoui is like Atoui64 but returns its result as a uint.
func Atoui(s string) (i uint, err os.Error) {
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

// Atoi is like Atoi64 but returns its result as an int.
func Atoi(s string) (i int, err os.Error) {
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

