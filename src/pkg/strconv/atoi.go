// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv
import (
	"os";
	"strconv"
)

type NumError struct {
	Num string;
	Error os.Error;
}

func (e *NumError) String() string {
	return "parsing " + e.Num + ": " + e.Error.String();
}


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
// The errors that Btoui64 returns have concrete type *NumError
// and include err.Num = s.  If s is empty or contains invalid
// digits, err.Error = os.EINVAL; if the value corresponding
// to s cannot be represented by a uint64, err.Error = os.ERANGE.
func Btoui64(s string, b int) (n uint64, err os.Error) {
	if b < 2 || b > 36 {
		err = os.ErrorString("invalid base " + Itoa(b));
		goto Error;
	}
	if len(s) < 1 {
		err = os.EINVAL;
		goto Error;
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
			n = 0;
			err = os.EINVAL;
			goto Error;
		}
		if int(v) >= b {
			n = 0;
			err = os.EINVAL;
			goto Error;
		}

		if n >= cutoff {
			// n*b overflows
			n = 1<<64-1;
			err = os.ERANGE;
			goto Error;
		}
		n *= uint64(b);

		n1 := n+uint64(v);
		if n1 < n {
			// n+v overflows
			n = 1<<64-1;
			err = os.ERANGE;
			goto Error;
		}
		n = n1;
	}

	return n, nil;

Error:
	return n, &NumError{s, err};
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
		return 0, &NumError{s, os.EINVAL}
	}

	// Look for octal, hex prefix.
	switch {
	case s[0] == '0' && len(s) > 1 && (s[1] == 'x' || s[1] == 'X'):
		n, err = Btoui64(s[2:len(s)], 16);
	case s[0] == '0':
		n, err = Btoui64(s, 8);
	default:
		n, err = Btoui64(s, 10);
	}

	if err != nil {
		err.(*NumError).Num = s;
	}
	return;
}


// Atoi64 is like Atoui64 but allows signed numbers and
// returns its result in an int64.
func Atoi64(s string) (i int64, err os.Error) {
	// Empty string bad.
	if len(s) == 0 {
		return 0, &NumError{s, os.EINVAL}
	}

	// Pick off leading sign.
	s0 := s;
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
	if err != nil && err.(*NumError).Error != os.ERANGE {
		err.(*NumError).Num = s0;
		return 0, err
	}
	if !neg && un >= 1<<63 {
		return 1<<63-1, &NumError{s0, os.ERANGE}
	}
	if neg && un > 1<<63 {
		return -1<<63, &NumError{s0, os.ERANGE}
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
	if e1 != nil && e1.(*NumError).Error != os.ERANGE {
		return 0, e1
	}
	i = uint(i1);
	if uint64(i) != i1 {
		return ^uint(0), &NumError{s, os.ERANGE}
	}
	return i, nil
}

// Atoi is like Atoi64 but returns its result as an int.
func Atoi(s string) (i int, err os.Error) {
	i1, e1 := Atoi64(s);
	if e1 != nil && e1.(*NumError).Error != os.ERANGE {
		return 0, e1
	}
	i = int(i1);
	if int64(i) != i1 {
		if i1 < 0 {
			return -1<<(intsize-1), &NumError{s, os.ERANGE}
		}
		return 1<<(intsize-1) - 1, &NumError{s, os.ERANGE}
	}
	return i, nil
}

