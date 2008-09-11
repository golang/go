// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package Integer

const ValueLen = 1000;
type Word uint32
type Value *[ValueLen]Word
export type IntegerImpl struct {
  val Value
}
export type Integer *IntegerImpl

const N = 4;
const H = 1
const L = 28;
const M = 1 << L - 1;


// ----------------------------------------------------------------------------
// Support

// TODO What are we going to about asserts?
func ASSERT(p bool) {
  if !p {
    panic("ASSERT failed");
  }
}


func CHECK(p bool) {
  if !p {
    panic("CHECK failed");
  }
}


func UNIMPLEMENTED(s string) {
  panic("UNIMPLEMENTED: ", s);
}


// ----------------------------------------------------------------------------
//

// TODO "len" is a reserved word at the moment - I think that's wrong
func len_(x Value) int {
  l := int(x[0]);
  if l < 0 { return -l; }
  return l;
}


func set_len(x Value, len_ int) {
  x[0] = Word(len_);
}


func alloc(len_ int) Value {
  ASSERT(len_ >= 0);
  z := new([ValueLen] Word);
  set_len(z, len_);
  return z;
}


func sign(x Value) bool {
  return int(x[0]) < 0;
}


func zero(x Value) bool {
  return x[0] == 0;
}


func neg(x Value) {
  x[0] = Word(-int(x[0]));
}


// ----------------------------------------------------------------------------
// Unsigned ops

func make(x int) Value;


func Update(x Word) (z, c Word) {
  // z = x & M;
  // c = x >> L;
  return x & M, x >> L;
}


func uadd(x, y Value) Value {
  xl := len_(x);
  yl := len_(y);
  if xl < yl {
    return uadd(y, x);
  }
  ASSERT(xl >= yl);
  z := alloc(xl + 1);

  i := 0;
  c := Word(0);
  for i < yl { z[i + H], c = Update(x[i + H] + y[i + H] + c); i++; }
  for i < xl { z[i + H], c = Update(x[i + H] + c); i++; }
  if c != 0 { z[i + H] = c; i++; }
  set_len(z, i);

  return z;
}


func usub(x, y Value) Value {
  xl := len_(x);
  yl := len_(y);
  if xl < yl {
    return uadd(y, x);
  }
  ASSERT(xl >= yl);
  z := alloc(xl + 1);

  i := 0;
  c := Word(0);
  for i < yl { z[i + H], c = Update(x[i + H] - y[i + H] + c); i++; }
  for i < xl { z[i + H], c = Update(x[i + H] + c); i++; }
  ASSERT(c == 0);  // usub(x, y) must be called with x >= y
  for i > 0 && z[i - 1 + H] == 0 { i--; }
  set_len(z, i);

  return z;
}


// Computes x = x*a + c (in place) for "small" a's.
func umul_add(x Value, a, c Word) Value {
  CHECK(0 <= a && a < (1 << N));
  CHECK(0 <= c && c < (1 << N));
  if (zero(x) || a == 0) && c == 0 {
    return make(0);
  }
  xl := len_(x);
  
  z := alloc(xl + 1);
  i := 0;
  for i < xl { z[i + H], c = Update(x[i + H] * a + c); i++; }
  if c != 0 { z[i + H] = c; i++; }  
  set_len(z, i);

  return z;
}


// Computes x = x div d (in place) for "small" d's. Returns x mod d.
func umod(x Value, d Word) Word {
  CHECK(0 < d && d < (1 << N));
  xl := len_(x);
  c := Word(0);

  i := xl;
  for i > 0 {
    i--;
    c = c << L + x[i + H];

    q := c / d;
    x[i + H] = q;
    
    //x[i + H] = c / d;  // BUG
    
    c = c % d;
  }
  if xl > 0 && x[xl - 1 + H] == 0 {
    set_len(x, xl - 1);
  }

  return c;
}


// Returns z = (x * y) div B, c = (x * y) mod B.
func umul1(x, y Word) (z Word, c Word) {
  const L2 = (L + 1) >> 1;
  const B2 = 1 << L2;
  const M2 = B2 - 1;
  
  x0 := x & M2;
  x1 := x >> L2;

  y0 := y & M2;
  y1 := y >> L2;
  
  z10 := x0*y0;
  z21 := x1*y0 + x0*y1 + (z10 >> L2);

  cc := x1*y1 + (z21 >> L2);  
  zz := ((z21 & M2) << L2) | (z10 & M2);
  return zz, cc
}


func umul(x Value, y Value) Value {
  if zero(x) || zero(y) {
    return make(0);
  }
  xl := len_(x);
  yl := len_(y);
  if xl < yl {
    return umul(y, x);  // for speed
  }
  ASSERT(xl >= yl && yl > 0);
  
  // initialize z
  zl := xl + yl;
  z := alloc(zl);
  for i := 0; i < zl; i++ { z[i + H] = 0; }
  
  k := 0;
  for j := 0; j < yl; j++ {
    d := y[j + H];
    if d != 0 {
      k = j;
      c := Word(0);
      for i := 0; i < xl; i++ {
        // compute z[k + H] += x[i + H] * d + c;
        t := z[k + H] + c;
        var z1 Word;
        z1, c = umul1(x[i + H], d);
        t += z1;
        z[k + H] = t & M;
        c += t >> L;
        k++;
      }
      if c != 0 {
        z[k + H] = Word(c);
        k++;
      }
    }
  }
  set_len(z, k);

  return z;
}


func ucmp(x Value, y Value) int {
  xl := len_(x);
  yl := len_(y);
  
  if xl != yl || xl == 0 {
    return xl - yl;
  }
  
  i := xl - 1;
  for i > 0 && x[i + H] == y[i + H] { i--; }
  return int(x[i + H]) - int(y[i + H]);
}


func ulog(x Value) int {
  xl := len_(x);
  if xl == 0 { return 0; }
  
  n := (xl - 1) * L;
  for t := x[xl - 1 + H]; t != 0; t >>= 1 { n++ };
  
  return n;
}


func make(x int) Value {
  if x == 0 {
    z := alloc(0);
    set_len(z, 0);
    return z;
  }
  
  if x == -x {  // smallest int
    z := alloc(2);
    z[0 + H] = 0;
    z[1 + H] = 1;
    set_len(z, -2);
    return z;
  }
  
  z := alloc(1);
  if x < 0 {
    z[0 + H] = Word(-x);
    set_len(z, -1);
  } else {
    z[0 + H] = Word(x);
    set_len(z, 1);
  }
  
  return z;
}


func make_from_string(s string) Value {
  // skip sign, if any
  i := 0;
  if len(s) > 0 && (s[i] == '-' || s[i] == '+') {
    i = 1;
  }
  
  // read digits
  x := make(0);
  for i < len(s) && '0' <= s[i] && s[i] <= '9' {
    x = umul_add(x, 10, Word(s[i] - '0'));
    i++;
  }
  
  // read sign
  if len(s) > 0 && s[0] == '-' {
    neg(x);
  }
  
  return x;
}


// ----------------------------------------------------------------------------
// Creation


func (x Integer) Init(val Value) Integer {
  x.val = val;
  return x;
}


// ----------------------------------------------------------------------------
// Signed ops

func add(x Value, y Value) Value {
  var z Value;
  if sign(x) == sign(y) {
    // x + y == x + y
    // (-x) + (-y) == -(x + y)
    z = uadd(x, y);
  } else {
    // x + (-y) == x - y == -(y - x)
    // (-x) + y == y - x == -(x - y)
    if ucmp(x, y) >= 0 {
      z = usub(x, y);
    } else {
      z = usub(y, x);
      neg(z);
    }
  }
  if sign(x) {
    neg(z);
  }
  return z;
}


func sub(x Value, y Value) Value {
  var z Value;
  if sign(x) != sign(y) {
    // x - (-y) == x + y
    // (-x) - y == -(x + y)
    z = uadd(x, y);
  } else {
    // x - y == x - y == -(y - x)
    // (-x) - (-y) == y - x == -(x - y)
    if ucmp(x, y) >= 0 {
      z = usub(x, y);
    } else {
      z = usub(y, x);
      neg(z);
    }
  }
  if sign(x) {
    neg(z);
  }
  return z;
}


func mul(x Value, y Value) Value {
  // x * y == x * y
  // x * (-y) == -(x * y)
  // (-x) * y == -(x * y)
  // (-x) * (-y) == x * y
  z := umul(x, y);
  if sign(x) != sign(y) {
    neg(z);
  }
  return z;
}


func mul_range(a, b int) Value {
  if a > b { return make(1) };
  if a == b { return make(a) };
  if a + 1 == b { return mul(make(a), make(b)) };
  m := (a + b) >> 1;
  ASSERT(a <= m && m < b);
  return mul(mul_range(a, m), mul_range(m + 1, b));
}


func fact(n int) Value {
  return mul_range(2, n);
}


// Returns a copy of x with space for one extra digit.
func copy(x Value) Value {
  xl := len_(x);
  
  z := alloc(xl + 1);  // add space for one extra digit
  for i := 0; i < xl; i++ { z[i + H] = x[i + H]; }
  set_len(z, int(x[0]));  // don't loose sign!
  
  return z;
}


func tostring(x Value) string {
  // allocate string
  // approx. length: 1 char for 3 bits
  n := ulog(x)/3 + 3;  // +1 (round up) +1 (sign) +1 (0 termination)
  //s := new([]byte, n);
  s := new([100000]byte);

  // convert
  z := copy(x);
  i := 0;
  for i == 0 || !zero(z) {
    s[i] = byte(umod(z, 10) + '0');
    i++;
  };
  if sign(x) {
    s[i] = '-';
    i++;
  }
  length := i;
  ASSERT(0 < i && i < n);
  
  // reverse in place
  i--;
  for j := 0; j < i; j++ {
    t := s[j];
    s[j] = s[i];
    s[i] = t;
    i--;
  }

  return string(s)[0:length];
}


// ----------------------------------------------------------------------------
// Creation

export func FromInt(v int) Integer {
  return new(IntegerImpl).Init(make(v));
}


export func FromString(s string) Integer {
  return new(IntegerImpl).Init(make_from_string(s));
}


// ----------------------------------------------------------------------------
// Arithmetic ops

func (x Integer) neg () Integer {
  if (zero(x.val)) {
    return new(IntegerImpl).Init(make(0));
  }
  
  z := copy(x.val);
  neg(z);
  return new(IntegerImpl).Init(z);
}


func (x Integer) add (y Integer) Integer {
  return new(IntegerImpl).Init(add(x.val, y.val));
}


func (x Integer) sub (y Integer) Integer {
  return new(IntegerImpl).Init(sub(x.val, y.val));
}


func (x Integer) mul (y Integer) Integer {
  return new(IntegerImpl).Init(mul(x.val, y.val));
}


func (x Integer) quo (y Integer) Integer {
  UNIMPLEMENTED("quo");
  return nil;
}


func (x Integer) rem (y Integer) Integer {
  UNIMPLEMENTED("rem");
  return nil;
}


func (x Integer) div (y Integer) Integer {
  UNIMPLEMENTED("div");
  return nil;
}


func (x Integer) mod (y Integer) Integer {
  UNIMPLEMENTED("mod");
  return nil;
}


// ----------------------------------------------------------------------------
// Arithmetic shifts

func (x Integer) shl (s uint) Integer {
  UNIMPLEMENTED("shl");
  return nil;
}


func (x Integer) shr (s uint) Integer {
  UNIMPLEMENTED("shr");
  return nil;
}


// ----------------------------------------------------------------------------
// Logical ops

func (x Integer) inv () Integer {
  UNIMPLEMENTED("inv");
  return nil;
}


func (x Integer) and (y Integer) Integer {
  UNIMPLEMENTED("and");
  return nil;
}


func (x Integer) or (y Integer) Integer {
  UNIMPLEMENTED("or");
  return nil;
}


func (x Integer) xor (y Integer) Integer {
  UNIMPLEMENTED("xor");
  return nil;
}


// ----------------------------------------------------------------------------
// Comparisons

func (x Integer) cmp (y Integer) int {
  // do better then this
  d := x.sub(y);
  switch {
    case sign(d.val): return -1;
    case zero(d.val): return  0;
    default         : return +1;
  }
  panic("UNREACHABLE");
}


func (x Integer) eql (y Integer) bool {
  return x.cmp(y) == 0;
}


func (x Integer) neq (y Integer) bool {
  return x.cmp(y) != 0;
}


func (x Integer) lss (y Integer) bool {
  return x.cmp(y) < 0;
}


func (x Integer) leq (y Integer) bool {
  return x.cmp(y) <= 0;
}


func (x Integer) gtr (y Integer) bool {
  return x.cmp(y) > 0;
}


func (x Integer) geq (y Integer) bool {
  return x.cmp(y) >= 0;
}


// ----------------------------------------------------------------------------
// Specials

export func Fact(n int) Integer {
  return new(IntegerImpl).Init(fact(n));
}


func (x Integer) ToString() string {
  return tostring(x.val);
}


func (x Integer) ToInt() int {
  v := x.val;
  if len_(v) <= 1 {
    if zero(v) {
      return 0;
    }
    i := int(v[0 + H]);
    if sign(v) {
      i = -i;  // incorrect for smallest int
    }
    return i;
  }
  panic("integer too large");
}
