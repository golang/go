// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements nat-to-string conversion functions.

package big

import (
	"errors"
	"fmt"
	"io"
	"math"
	"sync"
)

const digits = "0123456789abcdefghijklmnopqrstuvwxyz"

// Note: MaxBase = len(digits), but it must remain a rune constant
//       for API compatibility.

// MaxBase is the largest number base accepted for string conversions.
const MaxBase = 'z' - 'a' + 10 + 1

// maxPow returns (b**n, n) such that b**n is the largest power b**n <= _M.
// For instance maxPow(10) == (1e19, 19) for 19 decimal digits in a 64bit Word.
// In other words, at most n digits in base b fit into a Word.
// TODO(gri) replace this with a table, generated at build time.
func maxPow(b Word) (p Word, n int) {
	p, n = b, 1 // assuming b <= _M
	for max := _M / b; p <= max; {
		// p == b**n && p <= max
		p *= b
		n++
	}
	// p == b**n && p <= _M
	return
}

// pow returns x**n for n > 0, and 1 otherwise.
func pow(x Word, n int) (p Word) {
	// n == sum of bi * 2**i, for 0 <= i < imax, and bi is 0 or 1
	// thus x**n == product of x**(2**i) for all i where bi == 1
	// (Russian Peasant Method for exponentiation)
	p = 1
	for n > 0 {
		if n&1 != 0 {
			p *= x
		}
		x *= x
		n >>= 1
	}
	return
}

// scan scans the number corresponding to the longest possible prefix
// from r representing an unsigned number in a given conversion base.
// It returns the corresponding natural number res, the actual base b,
// a digit count, and a read or syntax error err, if any.
//
//	number   = [ prefix ] mantissa .
//	prefix   = "0" [ "x" | "X" | "b" | "B" ] .
//      mantissa = digits | digits "." [ digits ] | "." digits .
//	digits   = digit { digit } .
//	digit    = "0" ... "9" | "a" ... "z" | "A" ... "Z" .
//
// Unless fracOk is set, the base argument must be 0 or a value between
// 2 and MaxBase. If fracOk is set, the base argument must be one of
// 0, 2, 10, or 16. Providing an invalid base argument leads to a run-
// time panic.
//
// For base 0, the number prefix determines the actual base: A prefix of
// ``0x'' or ``0X'' selects base 16; if fracOk is not set, the ``0'' prefix
// selects base 8, and a ``0b'' or ``0B'' prefix selects base 2. Otherwise
// the selected base is 10 and no prefix is accepted.
//
// If fracOk is set, an octal prefix is ignored (a leading ``0'' simply
// stands for a zero digit), and a period followed by a fractional part
// is permitted. The result value is computed as if there were no period
// present; and the count value is used to determine the fractional part.
//
// A result digit count > 0 corresponds to the number of (non-prefix) digits
// parsed. A digit count <= 0 indicates the presence of a period (if fracOk
// is set, only), and -count is the number of fractional digits found.
// In this case, the actual value of the scanned number is res * b**count.
//
func (z nat) scan(r io.ByteScanner, base int, fracOk bool) (res nat, b, count int, err error) {
	// reject illegal bases
	baseOk := base == 0 ||
		!fracOk && 2 <= base && base <= MaxBase ||
		fracOk && (base == 2 || base == 10 || base == 16)
	if !baseOk {
		panic(fmt.Sprintf("illegal number base %d", base))
	}

	// one char look-ahead
	ch, err := r.ReadByte()
	if err != nil {
		return
	}

	// determine actual base
	b = base
	if base == 0 {
		// actual base is 10 unless there's a base prefix
		b = 10
		if ch == '0' {
			count = 1
			switch ch, err = r.ReadByte(); err {
			case nil:
				// possibly one of 0x, 0X, 0b, 0B
				if !fracOk {
					b = 8
				}
				switch ch {
				case 'x', 'X':
					b = 16
				case 'b', 'B':
					b = 2
				}
				switch b {
				case 16, 2:
					count = 0 // prefix is not counted
					if ch, err = r.ReadByte(); err != nil {
						// io.EOF is also an error in this case
						return
					}
				case 8:
					count = 0 // prefix is not counted
				}
			case io.EOF:
				// input is "0"
				res = z[:0]
				err = nil
				return
			default:
				// read error
				return
			}
		}
	}

	// convert string
	// Algorithm: Collect digits in groups of at most n digits in di
	// and then use mulAddWW for every such group to add them to the
	// result.
	z = z[:0]
	b1 := Word(b)
	bn, n := maxPow(b1) // at most n digits in base b1 fit into Word
	di := Word(0)       // 0 <= di < b1**i < bn
	i := 0              // 0 <= i < n
	dp := -1            // position of decimal point
	for {
		if fracOk && ch == '.' {
			fracOk = false
			dp = count
			// advance
			if ch, err = r.ReadByte(); err != nil {
				if err == io.EOF {
					err = nil
					break
				}
				return
			}
		}

		// convert rune into digit value d1
		var d1 Word
		switch {
		case '0' <= ch && ch <= '9':
			d1 = Word(ch - '0')
		case 'a' <= ch && ch <= 'z':
			d1 = Word(ch - 'a' + 10)
		case 'A' <= ch && ch <= 'Z':
			d1 = Word(ch - 'A' + 10)
		default:
			d1 = MaxBase + 1
		}
		if d1 >= b1 {
			r.UnreadByte() // ch does not belong to number anymore
			break
		}
		count++

		// collect d1 in di
		di = di*b1 + d1
		i++

		// if di is "full", add it to the result
		if i == n {
			z = z.mulAddWW(z, bn, di)
			di = 0
			i = 0
		}

		// advance
		if ch, err = r.ReadByte(); err != nil {
			if err == io.EOF {
				err = nil
				break
			}
			return
		}
	}

	if count == 0 {
		// no digits found
		switch {
		case base == 0 && b == 8:
			// there was only the octal prefix 0 (possibly followed by digits > 7);
			// count as one digit and return base 10, not 8
			count = 1
			b = 10
		case base != 0 || b != 8:
			// there was neither a mantissa digit nor the octal prefix 0
			err = errors.New("syntax error scanning number")
		}
		return
	}
	// count > 0

	// add remaining digits to result
	if i > 0 {
		z = z.mulAddWW(z, pow(b1, i), di)
	}
	res = z.norm()

	// adjust for fraction, if any
	if dp >= 0 {
		// 0 <= dp <= count > 0
		count = dp - count
	}

	return
}

// utoa converts x to an ASCII representation in the given base;
// base must be between 2 and MaxBase, inclusive.
func (x nat) utoa(base int) []byte {
	return x.itoa(false, base)
}

// itoa is like utoa but it prepends a '-' if neg && x != 0.
func (x nat) itoa(neg bool, base int) []byte {
	if base < 2 || base > MaxBase {
		panic("invalid base")
	}

	// x == 0
	if len(x) == 0 {
		return []byte("0")
	}
	// len(x) > 0

	// allocate buffer for conversion
	i := int(float64(x.bitLen())/math.Log2(float64(base))) + 1 // off by 1 at most
	if neg {
		i++
	}
	s := make([]byte, i)

	// convert power of two and non power of two bases separately
	if b := Word(base); b == b&-b {
		// shift is base b digit size in bits
		shift := trailingZeroBits(b) // shift > 0 because b >= 2
		mask := Word(1<<shift - 1)
		w := x[0]         // current word
		nbits := uint(_W) // number of unprocessed bits in w

		// convert less-significant words (include leading zeros)
		for k := 1; k < len(x); k++ {
			// convert full digits
			for nbits >= shift {
				i--
				s[i] = digits[w&mask]
				w >>= shift
				nbits -= shift
			}

			// convert any partial leading digit and advance to next word
			if nbits == 0 {
				// no partial digit remaining, just advance
				w = x[k]
				nbits = _W
			} else {
				// partial digit in current word w (== x[k-1]) and next word x[k]
				w |= x[k] << nbits
				i--
				s[i] = digits[w&mask]

				// advance
				w = x[k] >> (shift - nbits)
				nbits = _W - (shift - nbits)
			}
		}

		// convert digits of most-significant word w (omit leading zeros)
		for w != 0 {
			i--
			s[i] = digits[w&mask]
			w >>= shift
		}

	} else {
		bb, ndigits := maxPow(Word(b))

		// construct table of successive squares of bb*leafSize to use in subdivisions
		// result (table != nil) <=> (len(x) > leafSize > 0)
		table := divisors(len(x), b, ndigits, bb)

		// preserve x, create local copy for use by convertWords
		q := nat(nil).set(x)

		// convert q to string s in base b
		q.convertWords(s, b, ndigits, bb, table)

		// strip leading zeros
		// (x != 0; thus s must contain at least one non-zero digit
		// and the loop will terminate)
		i = 0
		for s[i] == '0' {
			i++
		}
	}

	if neg {
		i--
		s[i] = '-'
	}

	return s[i:]
}

// Convert words of q to base b digits in s. If q is large, it is recursively "split in half"
// by nat/nat division using tabulated divisors. Otherwise, it is converted iteratively using
// repeated nat/Word division.
//
// The iterative method processes n Words by n divW() calls, each of which visits every Word in the
// incrementally shortened q for a total of n + (n-1) + (n-2) ... + 2 + 1, or n(n+1)/2 divW()'s.
// Recursive conversion divides q by its approximate square root, yielding two parts, each half
// the size of q. Using the iterative method on both halves means 2 * (n/2)(n/2 + 1)/2 divW()'s
// plus the expensive long div(). Asymptotically, the ratio is favorable at 1/2 the divW()'s, and
// is made better by splitting the subblocks recursively. Best is to split blocks until one more
// split would take longer (because of the nat/nat div()) than the twice as many divW()'s of the
// iterative approach. This threshold is represented by leafSize. Benchmarking of leafSize in the
// range 2..64 shows that values of 8 and 16 work well, with a 4x speedup at medium lengths and
// ~30x for 20000 digits. Use nat_test.go's BenchmarkLeafSize tests to optimize leafSize for
// specific hardware.
//
func (q nat) convertWords(s []byte, b Word, ndigits int, bb Word, table []divisor) {
	// split larger blocks recursively
	if table != nil {
		// len(q) > leafSize > 0
		var r nat
		index := len(table) - 1
		for len(q) > leafSize {
			// find divisor close to sqrt(q) if possible, but in any case < q
			maxLength := q.bitLen()     // ~= log2 q, or at of least largest possible q of this bit length
			minLength := maxLength >> 1 // ~= log2 sqrt(q)
			for index > 0 && table[index-1].nbits > minLength {
				index-- // desired
			}
			if table[index].nbits >= maxLength && table[index].bbb.cmp(q) >= 0 {
				index--
				if index < 0 {
					panic("internal inconsistency")
				}
			}

			// split q into the two digit number (q'*bbb + r) to form independent subblocks
			q, r = q.div(r, q, table[index].bbb)

			// convert subblocks and collect results in s[:h] and s[h:]
			h := len(s) - table[index].ndigits
			r.convertWords(s[h:], b, ndigits, bb, table[0:index])
			s = s[:h] // == q.convertWords(s, b, ndigits, bb, table[0:index+1])
		}
	}

	// having split any large blocks now process the remaining (small) block iteratively
	i := len(s)
	var r Word
	if b == 10 {
		// hard-coding for 10 here speeds this up by 1.25x (allows for / and % by constants)
		for len(q) > 0 {
			// extract least significant, base bb "digit"
			q, r = q.divW(q, bb)
			for j := 0; j < ndigits && i > 0; j++ {
				i--
				// avoid % computation since r%10 == r - int(r/10)*10;
				// this appears to be faster for BenchmarkString10000Base10
				// and smaller strings (but a bit slower for larger ones)
				t := r / 10
				s[i] = '0' + byte(r-t<<3-t-t) // TODO(gri) replace w/ t*10 once compiler produces better code
				r = t
			}
		}
	} else {
		for len(q) > 0 {
			// extract least significant, base bb "digit"
			q, r = q.divW(q, bb)
			for j := 0; j < ndigits && i > 0; j++ {
				i--
				s[i] = digits[r%b]
				r /= b
			}
		}
	}

	// prepend high-order zeros
	for i > 0 { // while need more leading zeros
		i--
		s[i] = '0'
	}
}

// Split blocks greater than leafSize Words (or set to 0 to disable recursive conversion)
// Benchmark and configure leafSize using: go test -bench="Leaf"
//   8 and 16 effective on 3.0 GHz Xeon "Clovertown" CPU (128 byte cache lines)
//   8 and 16 effective on 2.66 GHz Core 2 Duo "Penryn" CPU
var leafSize int = 8 // number of Word-size binary values treat as a monolithic block

type divisor struct {
	bbb     nat // divisor
	nbits   int // bit length of divisor (discounting leading zeros) ~= log2(bbb)
	ndigits int // digit length of divisor in terms of output base digits
}

var cacheBase10 struct {
	sync.Mutex
	table [64]divisor // cached divisors for base 10
}

// expWW computes x**y
func (z nat) expWW(x, y Word) nat {
	return z.expNN(nat(nil).setWord(x), nat(nil).setWord(y), nil)
}

// construct table of powers of bb*leafSize to use in subdivisions
func divisors(m int, b Word, ndigits int, bb Word) []divisor {
	// only compute table when recursive conversion is enabled and x is large
	if leafSize == 0 || m <= leafSize {
		return nil
	}

	// determine k where (bb**leafSize)**(2**k) >= sqrt(x)
	k := 1
	for words := leafSize; words < m>>1 && k < len(cacheBase10.table); words <<= 1 {
		k++
	}

	// reuse and extend existing table of divisors or create new table as appropriate
	var table []divisor // for b == 10, table overlaps with cacheBase10.table
	if b == 10 {
		cacheBase10.Lock()
		table = cacheBase10.table[0:k] // reuse old table for this conversion
	} else {
		table = make([]divisor, k) // create new table for this conversion
	}

	// extend table
	if table[k-1].ndigits == 0 {
		// add new entries as needed
		var larger nat
		for i := 0; i < k; i++ {
			if table[i].ndigits == 0 {
				if i == 0 {
					table[0].bbb = nat(nil).expWW(bb, Word(leafSize))
					table[0].ndigits = ndigits * leafSize
				} else {
					table[i].bbb = nat(nil).mul(table[i-1].bbb, table[i-1].bbb)
					table[i].ndigits = 2 * table[i-1].ndigits
				}

				// optimization: exploit aggregated extra bits in macro blocks
				larger = nat(nil).set(table[i].bbb)
				for mulAddVWW(larger, larger, b, 0) == 0 {
					table[i].bbb = table[i].bbb.set(larger)
					table[i].ndigits++
				}

				table[i].nbits = table[i].bbb.bitLen()
			}
		}
	}

	if b == 10 {
		cacheBase10.Unlock()
	}

	return table
}
