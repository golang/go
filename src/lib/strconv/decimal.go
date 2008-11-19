// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Multiprecision decimal numbers.
// For floating-point formatting only; not general purpose.
// Only operations are assign and (binary) left/right shift.
// Can do binary floating point in multiprecision decimal precisely
// because 2 divides 10; cannot do decimal floating point
// in multiprecision binary precisely.

package strconv

package type Decimal struct {
	// TODO(rsc): Can make d[] a bit smaller and add
	// truncated bool;
	d [2000] byte;	// digits
	nd int;	// number of digits used
	dp int;	// decimal point
};
func (a *Decimal) String() string;
func (a *Decimal) Assign(v uint64);
func (a *Decimal) Shift(k int) *Decimal;
func (a *Decimal) Round(nd int) *Decimal;
func (a *Decimal) RoundUp(nd int) *Decimal;
func (a *Decimal) RoundDown(nd int) *Decimal;
func (a *Decimal) RoundedInteger() uint64;


func Copy(dst *[]byte, src *[]byte) int;
func DigitZero(dst *[]byte) int;

func (a *Decimal) String() string {
	n := 10 + a.nd;
	if a.dp > 0 {
		n += a.dp;
	}
	if a.dp < 0 {
		n += -a.dp;
	}

	buf := new([]byte, n);
	w := 0;
	switch {
	case a.nd == 0:
		return "0";

	case a.dp <= 0:
		// zeros fill space between decimal point and digits
		buf[w] = '0';
		w++;
		buf[w] = '.';
		w++;
		w += DigitZero(buf[w:w+-a.dp]);
		w += Copy(buf[w:w+a.nd], (&a.d)[0:a.nd]);

	case a.dp < a.nd:
		// decimal point in middle of digits
		w += Copy(buf[w:w+a.dp], (&a.d)[0:a.dp]);
		buf[w] = '.';
		w++;
		w += Copy(buf[w:w+a.nd-a.dp], (&a.d)[a.dp:a.nd]);

	default:
		// zeros fill space between digits and decimal point
		w += Copy(buf[w:w+a.nd], (&a.d)[0:a.nd]);
		w += DigitZero(buf[w:w+a.dp-a.nd]);
	}
	return string(buf[0:w]);
}

func Copy(dst *[]byte, src *[]byte) int {
	for i := 0; i < len(dst); i++ {
		dst[i] = src[i];
	}
	return len(dst);
}

func DigitZero(dst *[]byte) int {
	for i := 0; i < len(dst); i++ {
		dst[i] = '0';
	}
	return len(dst);
}

// Trim trailing zeros from number.
// (They are meaningless; the decimal point is tracked
// independent of the number of digits.)
func Trim(a *Decimal) {
	for a.nd > 0 && a.d[a.nd-1] == '0' {
		a.nd--;
	}
	if a.nd == 0 {
		a.dp = 0;
	}
}

// Assign v to a.
func (a *Decimal) Assign(v uint64) {
	var buf [50]byte;

	// Write reversed decimal in buf.
	n := 0;
	for v > 0 {
		v1 := v/10;
		v -= 10*v1;
		buf[n] = byte(v + '0');
		n++;
		v = v1;
	}

	// Reverse again to produce forward decimal in a.d.
	a.nd = 0;
	for n--; n>=0; n-- {
		a.d[a.nd] = buf[n];
		a.nd++;
	}
	a.dp = a.nd;
	Trim(a);
}

package func NewDecimal(i uint64) *Decimal {
	a := new(Decimal);
	a.Assign(i);
	return a;
}

// Maximum shift that we can do in one pass without overflow.
// Signed int has 31 bits, and we have to be able to accomodate 9<<k.
const MaxShift = 27

// Binary shift right (* 2) by k bits.  k <= MaxShift to avoid overflow.
func RightShift(a *Decimal, k uint) {
	r := 0;	// read pointer
	w := 0;	// write pointer

	// Pick up enough leading digits to cover first shift.
	n := 0;
	for ; n>>k == 0; r++ {
		if r >= a.nd {
			if n == 0 {
				// a == 0; shouldn't get here, but handle anyway.
				a.nd = 0;
				return;
			}
			for n>>k == 0 {
				n = n*10;
				r++;
			}
			break;
		}
		c := int(a.d[r]);
		n = n*10 + c-'0';
	}
	a.dp -= r-1;

	// Pick up a digit, put down a digit.
	for ; r < a.nd; r++ {
		c := int(a.d[r]);
		dig := n>>k;
		n -= dig<<k;
		a.d[w] = byte(dig+'0');
		w++;
		n = n*10 + c-'0';
	}

	// Put down extra digits.
	for n > 0 {
		dig := n>>k;
		n -= dig<<k;
		a.d[w] = byte(dig+'0');
		w++;
		n = n*10;
	}

	a.nd = w;
	Trim(a);
}

// Cheat sheet for left shift: table indexed by shift count giving
// number of new digits that will be introduced by that shift.
//
// For example, leftcheat[4] = {2, "625"}.  That means that
// if we are shifting by 4 (multiplying by 16), it will add 2 digits
// when the string prefix is "625" through "999", and one fewer digit
// if the string prefix is "000" through "624".
//
// Credit for this trick goes to Ken.

type LeftCheat struct {
	delta int;	// number of new digits
	cutoff string;	//   minus one digit if original < a.
}

var leftcheat = []LeftCheat {
	// Leading digits of 1/2^i = 5^i.
	// 5^23 is not an exact 64-bit floating point number,
	// so have to use bc for the math.
	/*
	seq 27 | sed 's/^/5^/' | bc |
	awk 'BEGIN{ print "\tLeftCheat{ 0, \"\" }," }
	{
		log2 = log(2)/log(10)
		printf("\tLeftCheat{ %d, \"%s\" },\t// * %d\n",
			int(log2*NR+1), $0, 2**NR)
	}'
	 */
	LeftCheat{ 0, "" },
	LeftCheat{ 1, "5" },	// * 2
	LeftCheat{ 1, "25" },	// * 4
	LeftCheat{ 1, "125" },	// * 8
	LeftCheat{ 2, "625" },	// * 16
	LeftCheat{ 2, "3125" },	// * 32
	LeftCheat{ 2, "15625" },	// * 64
	LeftCheat{ 3, "78125" },	// * 128
	LeftCheat{ 3, "390625" },	// * 256
	LeftCheat{ 3, "1953125" },	// * 512
	LeftCheat{ 4, "9765625" },	// * 1024
	LeftCheat{ 4, "48828125" },	// * 2048
	LeftCheat{ 4, "244140625" },	// * 4096
	LeftCheat{ 4, "1220703125" },	// * 8192
	LeftCheat{ 5, "6103515625" },	// * 16384
	LeftCheat{ 5, "30517578125" },	// * 32768
	LeftCheat{ 5, "152587890625" },	// * 65536
	LeftCheat{ 6, "762939453125" },	// * 131072
	LeftCheat{ 6, "3814697265625" },	// * 262144
	LeftCheat{ 6, "19073486328125" },	// * 524288
	LeftCheat{ 7, "95367431640625" },	// * 1048576
	LeftCheat{ 7, "476837158203125" },	// * 2097152
	LeftCheat{ 7, "2384185791015625" },	// * 4194304
	LeftCheat{ 7, "11920928955078125" },	// * 8388608
	LeftCheat{ 8, "59604644775390625" },	// * 16777216
	LeftCheat{ 8, "298023223876953125" },	// * 33554432
	LeftCheat{ 8, "1490116119384765625" },	// * 67108864
	LeftCheat{ 9, "7450580596923828125" },	// * 134217728
}

// Is the leading prefix of b lexicographically less than s?
func PrefixIsLessThan(b *[]byte, s string) bool {
	for i := 0; i < len(s); i++ {
		if i >= len(b) {
			return true;
		}
		if b[i] != s[i] {
			return b[i] < s[i];
		}
	}
	return false;
}

// Binary shift left (/ 2) by k bits.  k <= MaxShift to avoid overflow.
func LeftShift(a *Decimal, k uint) {
	delta := leftcheat[k].delta;
	if PrefixIsLessThan((&a.d)[0:a.nd], leftcheat[k].cutoff) {
		delta--;
	}

	r := a.nd;	// read index
	w := a.nd + delta;	// write index
	n := 0;

	// Pick up a digit, put down a digit.
	for r--; r >= 0; r-- {
		n += (int(a.d[r])-'0') << k;
		quo := n/10;
		rem := n - 10*quo;
		w--;
		a.d[w] = byte(rem+'0');
		n = quo;
	}

	// Put down extra digits.
	for n > 0 {
		quo := n/10;
		rem := n - 10*quo;
		w--;
		a.d[w] = byte(rem+'0');
		n = quo;
	}

	if w != 0 {
		// TODO: Remove - has no business panicking.
		panicln("strconv: bad LeftShift", w);
	}
	a.nd += delta;
	a.dp += delta;
	Trim(a);
}

// Binary shift left (k > 0) or right (k < 0).
// Returns receiver for convenience.
func (a *Decimal) Shift(k int) *Decimal {
	switch {
	case a.nd == 0:
		// nothing to do: a == 0
	case k > 0:
		for k > MaxShift {
			LeftShift(a, MaxShift);
			k -= MaxShift;
		}
		LeftShift(a, uint(k));
	case k < 0:
		for k < -MaxShift {
			RightShift(a, MaxShift);
			k += MaxShift;
		}
		RightShift(a, uint(-k));
	}
	return a;
}

// If we chop a at nd digits, should we round up?
func ShouldRoundUp(a *Decimal, nd int) bool {
	if nd <= 0 || nd >= a.nd {
		return false;
	}
	if a.d[nd] == '5' && nd+1 == a.nd {	// exactly halfway - round to even
		return (a.d[nd-1] - '0') % 2 != 0;
	}
	// not halfway - digit tells all
	return a.d[nd] >= '5';
}

// Round a to nd digits (or fewer).
// Returns receiver for convenience.
func (a *Decimal) Round(nd int) *Decimal {
	if nd <= 0 || nd >= a.nd {
		return a;
	}
	if(ShouldRoundUp(a, nd)) {
		return a.RoundUp(nd);
	}
	return a.RoundDown(nd);
}

// Round a down to nd digits (or fewer).
// Returns receiver for convenience.
func (a *Decimal) RoundDown(nd int) *Decimal {
	if nd <= 0 || nd >= a.nd {
		return a;
	}
	a.nd = nd;
	Trim(a);
	return a;
}

// Round a up to nd digits (or fewer).
// Returns receiver for convenience.
func (a *Decimal) RoundUp(nd int) *Decimal {
	if nd <= 0 || nd >= a.nd {
		return a;
	}

	// round up
	for i := nd-1; i >= 0; i-- {
		c := a.d[i];
		if c < '9' {	 // can stop after this digit
			a.d[i]++;
			a.nd = i+1;
			return a;
		}
	}

	// Number is all 9s.
	// Change to single 1 with adjusted decimal point.
	a.d[0] = '1';
	a.nd = 1;
	a.dp++;
	return a;
}

// Extract integer part, rounded appropriately.
// No guarantees about overflow.
func (a *Decimal) RoundedInteger() uint64 {
	if a.dp > 20 {
		return 0xFFFFFFFFFFFFFFFF;
	}
	var i int;
	n := uint64(0);
	for i = 0; i < a.dp && i < a.nd; i++ {
		n = n*10 + uint64(a.d[i] - '0');
	}
	for ; i < a.dp; i++ {
		n *= 10;
	}
	if ShouldRoundUp(a, a.dp) {
		n++;
	}
	return n;
}

