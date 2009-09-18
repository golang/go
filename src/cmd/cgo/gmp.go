// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
An example of wrapping a C library in Go. This is the GNU
multiprecision library gmp's integer type mpz_t wrapped to look like
the Go package big's integer type Int.

This is a syntactically valid Go program—it can be parsed with the Go
parser and processed by godoc—but it is not compiled directly by 6g.
Instead, a separate tool, cgo, processes it to produce three output
files.  The first two, 6g.go and 6c.c, are a Go source file for 6g and
a C source file for 6c; both compile as part of the named package
(gmp, in this example).  The third, gcc.c, is a C source file for gcc;
it compiles into a shared object (.so) that is dynamically linked into
any 6.out that imports the first two files.

The stanza

	// #include <gmp.h>
	import "C"

is a signal to cgo.  The doc comment on the import of "C" provides
additional context for the C file.  Here it is just a single #include
but it could contain arbitrary C definitions to be imported and used.

Cgo recognizes any use of a qualified identifier C.xxx and uses gcc to
find the definition of xxx.  If xxx is a type, cgo replaces C.xxx with
a Go translation.  C arithmetic types translate to precisely-sized Go
arithmetic types.  A C struct translates to a Go struct, field by
field; unrepresentable fields are replaced with opaque byte arrays.  A
C union translates into a struct containing the first union member and
perhaps additional padding.  C arrays become Go arrays.  C pointers
become Go pointers.  C function pointers and void pointers become Go's
*byte.

For example, mpz_t is defined in <gmp.h> as:

	typedef unsigned long int mp_limb_t;

	typedef struct
	{
		int _mp_alloc;
		int _mp_size;
		mp_limb_t *_mp_d;
	} __mpz_struct;

	typedef __mpz_struct mpz_t[1];

Cgo generates:

	type _C_int int32
	type _C_mp_limb_t uint64
	type _C___mpz_struct struct {
		_mp_alloc _C_int;
		_mp_size _C_int;
		_mp_d *_C_mp_limb_t;
	}
	type _C_mpz_t [1]_C___mpz_struct

and then replaces each occurrence of a type C.xxx with _C_xxx.

If xxx is data, cgo arranges for C.xxx to refer to the C variable,
with the type translated as described above.  To do this, cgo must
introduce a Go variable that points at the C variable (the linker can
be told to initialize this pointer).  For example, if the gmp library
provided

	mpz_t zero;

then cgo would rewrite a reference to C.zero by introducing

	var _C_zero *C.mpz_t

and then replacing all instances of C.zero with (*_C_zero).

Cgo's most interesting translation is for functions.  If xxx is a C
function, then cgo rewrites C.xxx into a new function _C_xxx that
calls the C xxx in a standard pthread.  The new function translates
its arguments, calls xxx, and translates the return value.

Translation of parameters and the return value follows the type
translation above with one extension: a function expecting a char*
will change to expect a string, and a function returning a char* will
change to return a string.  The wrapper that cgo generates for the
first case allocates a new C string, passes that pointer to the C
function, and then frees the string when the function returns.  The
wrapper for the second case assumes the char* being returned is
pointer that must be freed.  It makes a Go string with a copy of the
contents and then frees the pointer.  The char* conventions are a
useful heuristic; there should be some way to override them but isn't
yet.  One can also imagine wrapping Go functions being passed into C
functions so that C can call them.

Garbage collection is the big problem.  It is fine for the Go world to
have pointers into the C world and to free those pointers when they
are no longer needed.  To help, the garbage collector calls an
object's destroy() method prior to collecting it.  C pointers can be
wrapped by Go objects with appropriate destroy methods.

It is much more difficult for the C world to have pointers into the Go
world, because the Go garbage collector is unaware of the memory
allocated by C. I think the most important consideration is not to
constrain future implementations, so the rule is basically that Go
code can hand a Go pointer to C code but must separately arrange for
Go to hang on to a reference to the pointer until C is done with it.

Note: the sketches assume that the char* <-> string conversions described
above have been thrown away.  Otherwise one can't pass nil as the first
argument to mpz_get_str.

Sketch of 6c.c:

	// NOTE: Maybe cgo is smart enough to figure out that
	// mpz_init's real C name is __gmpz_init and use that instead.

	// Tell dynamic linker to initialize _cgo_mpz_init in this file
	// to point at the function of the same name in gcc.c.
	#pragma dynld _cgo_mpz_init _cgo_mpz_init "gmp.so"
	#pragma dynld _cgo_mpz_get_str _cgo_mpz_get_str "gmp.so"

	void (*_cgo_mpz_init)(void*);
	void (*_cgo_mpz_get_str)(void*);

	// implementation of Go function called as C.mpz_init below.
	void
	gmp·_C_mpz_init(struct { char x[8]; } p)	// dummy struct, same size as 6g parameter frame
	{
		cgocall(_cgo_mpz_init, &p);
	}

	void
	gmp·_C_mpz_get_str(struct { char x[32]; } p)
	{
		cgocall(_cgo_mpz_get_str, &p);
	}

Sketch of 6g.go:

	// Type declarations from above, omitted.

	// Extern declarations for 6c.c functions
	func _C_mpz_init(*_C_mpz_t)
	func _C_mpz_get_str(*_C_char, int32, *_C_mpz_t) *_C_char

	// Original Go source with C.xxx replaced by _C_xxx
	// as described above.

Sketch of gcc.c:

	void
	_cgo_mpz_init(void *v)
	{
		struct {
			__mpz_struct *p1;	// not mpz_t because of C array passing rule
		} *a = v;
		mpz_init(a->p1);
	}

	void
	_cgo_mpz_get_str(void *v)
	{
		struct {
			char *p1;
			int32 p2;
			in32 _pad1;
			__mpz_struct *p3;
			char *p4;
		} *a = v;
		a->p4 = mpz_get_str(a->p1, a->p2, a->p3);
	}

Gmp defines mpz_t as __mpz_struct[1], meaning that if you
declare one it takes up a struct worth of space, but when you
pass one to a function, it passes a pointer to the space instead
of copying it.  This can't be modeled directly in Go or in C structs
so some rewriting happens in the generated files.  In Go,
the functions take *_C_mpz_t instead of _C_mpz_t, and in the
GCC structs, the parameters are __mpz_struct* instead of mpz_t.

*/

package gmp

// #include <gmp.h>
import "C"


/*
 * one of a kind
 */

// An Int represents a signed multi-precision integer.
// The zero value for an Int represents the value 0.
type Int struct {
	i C.mpz_t;
	init bool;
}

// NewInt returns a new Int initialized to x.
func NewInt(x int64) *Int {
	z := new(Int);
	z.init = true;
	C.mpz_init(&z.i);
	C.mpz_set(&z.i, x);
	return z;
}

// Int promises that the zero value is a 0, but in gmp
// the zero value is a crash.  To bridge the gap, the
// init bool says whether this is a valid gmp value.
// doinit initializes z.i if it needs it.  This is not inherent
// to FFI, just a mismatch between Go's convention of
// making zero values useful and gmp's decision not to.
func (z *Int) doinit() {
	if z.init {
		return;
	}
	z.init = true;
	C.mpz_init(&z.i);
}

// Bytes returns z's representation as a big-endian byte array.
func (z *Int) Bytes() []byte {
	b := make([]byte, (z.Len() + 7) / 8);
	n := C.size_t(len(b));
	C.mpz_export(&b[0], &n, 1, 1, 1, 0, &z.i);
	return b[0:n];
}

// Len returns the length of z in bits.  0 is considered to have length 1.
func (z *Int) Len() int {
	z.doinit();
	return int(C.mpz_sizeinbase(&z.i, 2));
}

// Set sets z = x and returns z.
func (z *Int) Set(x *Int) *Int {
	z.doinit();
	C.mpz_set(&z.i, x);
	return z;
}

// SetBytes interprets b as the bytes of a big-endian integer
// and sets z to that value.
func (z *Int) SetBytes(b []byte) *Int {
	z.doinit();
	if len(b) == 0 {
		z.SetInt64(0);
	} else {
		C.mpz_import(&z.i, len(b), 1, 1, 1, 0, &b[0]);
	}
	return z;
}

// SetInt64 sets z = x and returns z.
func (z *Int) SetInt64(x int64) *Int {
	z.doinit();
	// TODO(rsc): more work on 32-bit platforms
	C.mpz_set_si(z, x);
	return z;
}

// SetString interprets s as a number in the given base
// and sets z to that value.  The base must be in the range [2,36].
// SetString returns an error if s cannot be parsed or the base is invalid.
func (z *Int) SetString(s string, base int) os.Error {
	z.doinit();
	if base < 2 || base > 36 {
		return os.EINVAL;
	}
	if C.mpz_set_str(&z.i, s, base) < 0 {
		return os.EINVAL;
	}
	return z;
}

// String returns the decimal representation of z.
func (z *Int) String() string {
	z.doinit();
	return C.mpz_get_str(nil, 10, &z.i);
}

func (z *Int) destroy() {
	if z.init {
		C.mpz_clear(z);
	}
	z.init = false;
}


/*
 * arithmetic
 */

// Add sets z = x + y and returns z.
func (z *Int) Add(x, y *Int) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_add(&z.i, &x.i, &y.i);
	return z;
}

// Sub sets z = x - y and returns z.
func (z *Int) Sub(x, y *Int) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_sub(&z.i, &x.i, &y.i);
	return z;
}

// Mul sets z = x * y and returns z.
func (z *Int) Mul(x, y *Int) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_mul(&z.i, &x.i, &y.i);
	return z;
}

// Div sets z = x / y, rounding toward zero, and returns z.
func (z *Int) Div(x, y *Int) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_tdiv_q(&z.i, &x.i, &y.i);
	return z;
}

// Mod sets z = x % y and returns z.
// XXX Unlike in Go, the result is always positive.
func (z *Int) Mod(x, y *Int) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_tdiv_r(&z.i, &x.i, &y.i);
	return z;
}

// Lsh sets z = x << s and returns z.
func (z *Int) Lsh(x *Int, s uint) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_mul_2exp(&z.i, &x.i, s);
}

// Rsh sets z = x >> s and returns z.
func (z *Int) Rsh(x *int, s uint) *Int {
	x.doinit();
	y.doinit();
	z.doinit();
	C.mpz_div_2exp(&z.i, &x.i, s);
}

// Exp sets z = x^y % m and returns z.
// If m == nil, Exp sets z = x^y.
func (z *Int) Exp(x, y, m *Int) *Int {
	m.doinit();
	x.doinit();
	y.doinit();
	z.doinit();
	if m == nil {
		C.mpz_pow_ui(&z.i, &x.i, mpz_get_ui(&y.i));
	} else {
		C.mpz_powm(&z.i, &x.i, &y.i, &m.i);
	}
	return z;
}

// Neg sets z = -x and returns z.
func (z *Int) Neg(x *Int) *Int {
	x.doinit();
	z.doinit();
	C.mpz_neg(&z.i, &x.i);
	return z;
}

// Abs sets z to the absolute value of x and returns z.
func (z *Int) Abs(x *Int) *Int {
	x.doinit();
	z.doinit();
	C.mpz_abs(&z.i, &x.i);
	return z;
}


/*
 * functions without a clear receiver
 */

// CmpInt compares x and y. The result is
//
//   -1 if x <  y
//    0 if x == y
//   +1 if x >  y
//
func CmpInt(x, y *Int) int {
	x.doinit();
	y.doinit();
	return C.mpz_cmp(&x.i, &y.i);
}

// DivModInt sets q = x / y and r = x % y.
func DivModInt(q, r, x, y *Int) {
	q.doinit();
	r.doinit();
	x.doinit();
	y.doinit();
	C.mpz_tdiv_qr(&q.i, &r.i, &x.i, &y.i);
}

// GcdInt sets d to the greatest common divisor of a and b,
// which must be positive numbers.
// If x and y are not nil, GcdInt sets x and y such that d = a*x + b*y.
// If either a or b is not positive, GcdInt sets d = x = y = 0.
func GcdInt(d, x, y, a, b *Int) {
	d.doinit();
	x.doinit();
	y.doinit();
	a.doinit();
	b.doinit();
	C.mpz_gcdext(&d.i, &x.i, &y.i, &a.i, &b.i);
}
