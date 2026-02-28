// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test concurrency primitives: power series.

// Like powser1.go but uses channels of interfaces.
// Has not been cleaned up as much as powser1.go, to keep
// it distinct and therefore a different test.

// Power series package
// A power series is a channel, along which flow rational
// coefficients.  A denominator of zero signifies the end.
// Original code in Newsqueak by Doug McIlroy.
// See Squinting at Power Series by Doug McIlroy,
//   https://swtch.com/~rsc/thread/squint.pdf

package main

import "os"

type rat struct {
	num, den int64 // numerator, denominator
}

type item interface {
	pr()
	eq(c item) bool
}

func (u *rat) pr() {
	if u.den == 1 {
		print(u.num)
	} else {
		print(u.num, "/", u.den)
	}
	print(" ")
}

func (u *rat) eq(c item) bool {
	c1 := c.(*rat)
	return u.num == c1.num && u.den == c1.den
}

type dch struct {
	req chan int
	dat chan item
	nam int
}

type dch2 [2]*dch

var chnames string
var chnameserial int
var seqno int

func mkdch() *dch {
	c := chnameserial % len(chnames)
	chnameserial++
	d := new(dch)
	d.req = make(chan int)
	d.dat = make(chan item)
	d.nam = c
	return d
}

func mkdch2() *dch2 {
	d2 := new(dch2)
	d2[0] = mkdch()
	d2[1] = mkdch()
	return d2
}

// split reads a single demand channel and replicates its
// output onto two, which may be read at different rates.
// A process is created at first demand for an item and dies
// after the item has been sent to both outputs.

// When multiple generations of split exist, the newest
// will service requests on one channel, which is
// always renamed to be out[0]; the oldest will service
// requests on the other channel, out[1].  All generations but the
// newest hold queued data that has already been sent to
// out[0].  When data has finally been sent to out[1],
// a signal on the release-wait channel tells the next newer
// generation to begin servicing out[1].

func dosplit(in *dch, out *dch2, wait chan int) {
	both := false // do not service both channels

	select {
	case <-out[0].req:

	case <-wait:
		both = true
		select {
		case <-out[0].req:

		case <-out[1].req:
			out[0], out[1] = out[1], out[0]
		}
	}

	seqno++
	in.req <- seqno
	release := make(chan int)
	go dosplit(in, out, release)
	dat := <-in.dat
	out[0].dat <- dat
	if !both {
		<-wait
	}
	<-out[1].req
	out[1].dat <- dat
	release <- 0
}

func split(in *dch, out *dch2) {
	release := make(chan int)
	go dosplit(in, out, release)
	release <- 0
}

func put(dat item, out *dch) {
	<-out.req
	out.dat <- dat
}

func get(in *dch) *rat {
	seqno++
	in.req <- seqno
	return (<-in.dat).(*rat)
}

// Get one item from each of n demand channels

func getn(in []*dch) []item {
	n := len(in)
	if n != 2 {
		panic("bad n in getn")
	}
	req := make([]chan int, 2)
	dat := make([]chan item, 2)
	out := make([]item, 2)
	var i int
	var it item
	for i = 0; i < n; i++ {
		req[i] = in[i].req
		dat[i] = nil
	}
	for n = 2 * n; n > 0; n-- {
		seqno++

		select {
		case req[0] <- seqno:
			dat[0] = in[0].dat
			req[0] = nil
		case req[1] <- seqno:
			dat[1] = in[1].dat
			req[1] = nil
		case it = <-dat[0]:
			out[0] = it
			dat[0] = nil
		case it = <-dat[1]:
			out[1] = it
			dat[1] = nil
		}
	}
	return out
}

// Get one item from each of 2 demand channels

func get2(in0 *dch, in1 *dch) []item {
	return getn([]*dch{in0, in1})
}

func copy(in *dch, out *dch) {
	for {
		<-out.req
		out.dat <- get(in)
	}
}

func repeat(dat item, out *dch) {
	for {
		put(dat, out)
	}
}

type PS *dch    // power series
type PS2 *[2]PS // pair of power series

var Ones PS
var Twos PS

func mkPS() *dch {
	return mkdch()
}

func mkPS2() *dch2 {
	return mkdch2()
}

// Conventions
// Upper-case for power series.
// Lower-case for rationals.
// Input variables: U,V,...
// Output variables: ...,Y,Z

// Integer gcd; needed for rational arithmetic

func gcd(u, v int64) int64 {
	if u < 0 {
		return gcd(-u, v)
	}
	if u == 0 {
		return v
	}
	return gcd(v%u, u)
}

// Make a rational from two ints and from one int

func i2tor(u, v int64) *rat {
	g := gcd(u, v)
	r := new(rat)
	if v > 0 {
		r.num = u / g
		r.den = v / g
	} else {
		r.num = -u / g
		r.den = -v / g
	}
	return r
}

func itor(u int64) *rat {
	return i2tor(u, 1)
}

var zero *rat
var one *rat

// End mark and end test

var finis *rat

func end(u *rat) int64 {
	if u.den == 0 {
		return 1
	}
	return 0
}

// Operations on rationals

func add(u, v *rat) *rat {
	g := gcd(u.den, v.den)
	return i2tor(u.num*(v.den/g)+v.num*(u.den/g), u.den*(v.den/g))
}

func mul(u, v *rat) *rat {
	g1 := gcd(u.num, v.den)
	g2 := gcd(u.den, v.num)
	r := new(rat)
	r.num = (u.num / g1) * (v.num / g2)
	r.den = (u.den / g2) * (v.den / g1)
	return r
}

func neg(u *rat) *rat {
	return i2tor(-u.num, u.den)
}

func sub(u, v *rat) *rat {
	return add(u, neg(v))
}

func inv(u *rat) *rat { // invert a rat
	if u.num == 0 {
		panic("zero divide in inv")
	}
	return i2tor(u.den, u.num)
}

// print eval in floating point of PS at x=c to n terms
func Evaln(c *rat, U PS, n int) {
	xn := float64(1)
	x := float64(c.num) / float64(c.den)
	val := float64(0)
	for i := 0; i < n; i++ {
		u := get(U)
		if end(u) != 0 {
			break
		}
		val = val + x*float64(u.num)/float64(u.den)
		xn = xn * x
	}
	print(val, "\n")
}

// Print n terms of a power series
func Printn(U PS, n int) {
	done := false
	for ; !done && n > 0; n-- {
		u := get(U)
		if end(u) != 0 {
			done = true
		} else {
			u.pr()
		}
	}
	print(("\n"))
}

func Print(U PS) {
	Printn(U, 1000000000)
}

// Evaluate n terms of power series U at x=c
func eval(c *rat, U PS, n int) *rat {
	if n == 0 {
		return zero
	}
	y := get(U)
	if end(y) != 0 {
		return zero
	}
	return add(y, mul(c, eval(c, U, n-1)))
}

// Power-series constructors return channels on which power
// series flow.  They start an encapsulated generator that
// puts the terms of the series on the channel.

// Make a pair of power series identical to a given power series

func Split(U PS) *dch2 {
	UU := mkdch2()
	go split(U, UU)
	return UU
}

// Add two power series
func Add(U, V PS) PS {
	Z := mkPS()
	go func(U, V, Z PS) {
		var uv []item
		for {
			<-Z.req
			uv = get2(U, V)
			switch end(uv[0].(*rat)) + 2*end(uv[1].(*rat)) {
			case 0:
				Z.dat <- add(uv[0].(*rat), uv[1].(*rat))
			case 1:
				Z.dat <- uv[1]
				copy(V, Z)
			case 2:
				Z.dat <- uv[0]
				copy(U, Z)
			case 3:
				Z.dat <- finis
			}
		}
	}(U, V, Z)
	return Z
}

// Multiply a power series by a constant
func Cmul(c *rat, U PS) PS {
	Z := mkPS()
	go func(c *rat, U, Z PS) {
		done := false
		for !done {
			<-Z.req
			u := get(U)
			if end(u) != 0 {
				done = true
			} else {
				Z.dat <- mul(c, u)
			}
		}
		Z.dat <- finis
	}(c, U, Z)
	return Z
}

// Subtract

func Sub(U, V PS) PS {
	return Add(U, Cmul(neg(one), V))
}

// Multiply a power series by the monomial x^n

func Monmul(U PS, n int) PS {
	Z := mkPS()
	go func(n int, U PS, Z PS) {
		for ; n > 0; n-- {
			put(zero, Z)
		}
		copy(U, Z)
	}(n, U, Z)
	return Z
}

// Multiply by x

func Xmul(U PS) PS {
	return Monmul(U, 1)
}

func Rep(c *rat) PS {
	Z := mkPS()
	go repeat(c, Z)
	return Z
}

// Monomial c*x^n

func Mon(c *rat, n int) PS {
	Z := mkPS()
	go func(c *rat, n int, Z PS) {
		if c.num != 0 {
			for ; n > 0; n = n - 1 {
				put(zero, Z)
			}
			put(c, Z)
		}
		put(finis, Z)
	}(c, n, Z)
	return Z
}

func Shift(c *rat, U PS) PS {
	Z := mkPS()
	go func(c *rat, U, Z PS) {
		put(c, Z)
		copy(U, Z)
	}(c, U, Z)
	return Z
}

// simple pole at 1: 1/(1-x) = 1 1 1 1 1 ...

// Convert array of coefficients, constant term first
// to a (finite) power series

/*
func Poly(a [] *rat) PS{
	Z:=mkPS()
	begin func(a [] *rat, Z PS){
		j:=0
		done:=0
		for j=len(a); !done&&j>0; j=j-1)
			if(a[j-1].num!=0) done=1
		i:=0
		for(; i<j; i=i+1) put(a[i],Z)
		put(finis,Z)
	}()
	return Z
}
*/

// Multiply. The algorithm is
//	let U = u + x*UU
//	let V = v + x*VV
//	then UV = u*v + x*(u*VV+v*UU) + x*x*UU*VV

func Mul(U, V PS) PS {
	Z := mkPS()
	go func(U, V, Z PS) {
		<-Z.req
		uv := get2(U, V)
		if end(uv[0].(*rat)) != 0 || end(uv[1].(*rat)) != 0 {
			Z.dat <- finis
		} else {
			Z.dat <- mul(uv[0].(*rat), uv[1].(*rat))
			UU := Split(U)
			VV := Split(V)
			W := Add(Cmul(uv[0].(*rat), VV[0]), Cmul(uv[1].(*rat), UU[0]))
			<-Z.req
			Z.dat <- get(W)
			copy(Add(W, Mul(UU[1], VV[1])), Z)
		}
	}(U, V, Z)
	return Z
}

// Differentiate

func Diff(U PS) PS {
	Z := mkPS()
	go func(U, Z PS) {
		<-Z.req
		u := get(U)
		if end(u) == 0 {
			done := false
			for i := 1; !done; i++ {
				u = get(U)
				if end(u) != 0 {
					done = true
				} else {
					Z.dat <- mul(itor(int64(i)), u)
					<-Z.req
				}
			}
		}
		Z.dat <- finis
	}(U, Z)
	return Z
}

// Integrate, with const of integration
func Integ(c *rat, U PS) PS {
	Z := mkPS()
	go func(c *rat, U, Z PS) {
		put(c, Z)
		done := false
		for i := 1; !done; i++ {
			<-Z.req
			u := get(U)
			if end(u) != 0 {
				done = true
			}
			Z.dat <- mul(i2tor(1, int64(i)), u)
		}
		Z.dat <- finis
	}(c, U, Z)
	return Z
}

// Binomial theorem (1+x)^c

func Binom(c *rat) PS {
	Z := mkPS()
	go func(c *rat, Z PS) {
		n := 1
		t := itor(1)
		for c.num != 0 {
			put(t, Z)
			t = mul(mul(t, c), i2tor(1, int64(n)))
			c = sub(c, one)
			n++
		}
		put(finis, Z)
	}(c, Z)
	return Z
}

// Reciprocal of a power series
//	let U = u + x*UU
//	let Z = z + x*ZZ
//	(u+x*UU)*(z+x*ZZ) = 1
//	z = 1/u
//	u*ZZ + z*UU +x*UU*ZZ = 0
//	ZZ = -UU*(z+x*ZZ)/u

func Recip(U PS) PS {
	Z := mkPS()
	go func(U, Z PS) {
		ZZ := mkPS2()
		<-Z.req
		z := inv(get(U))
		Z.dat <- z
		split(Mul(Cmul(neg(z), U), Shift(z, ZZ[0])), ZZ)
		copy(ZZ[1], Z)
	}(U, Z)
	return Z
}

// Exponential of a power series with constant term 0
// (nonzero constant term would make nonrational coefficients)
// bug: the constant term is simply ignored
//	Z = exp(U)
//	DZ = Z*DU
//	integrate to get Z

func Exp(U PS) PS {
	ZZ := mkPS2()
	split(Integ(one, Mul(ZZ[0], Diff(U))), ZZ)
	return ZZ[1]
}

// Substitute V for x in U, where the leading term of V is zero
//	let U = u + x*UU
//	let V = v + x*VV
//	then S(U,V) = u + VV*S(V,UU)
// bug: a nonzero constant term is ignored

func Subst(U, V PS) PS {
	Z := mkPS()
	go func(U, V, Z PS) {
		VV := Split(V)
		<-Z.req
		u := get(U)
		Z.dat <- u
		if end(u) == 0 {
			if end(get(VV[0])) != 0 {
				put(finis, Z)
			} else {
				copy(Mul(VV[0], Subst(U, VV[1])), Z)
			}
		}
	}(U, V, Z)
	return Z
}

// Monomial Substitution: U(c x^n)
// Each Ui is multiplied by c^i and followed by n-1 zeros

func MonSubst(U PS, c0 *rat, n int) PS {
	Z := mkPS()
	go func(U, Z PS, c0 *rat, n int) {
		c := one
		for {
			<-Z.req
			u := get(U)
			Z.dat <- mul(u, c)
			c = mul(c, c0)
			if end(u) != 0 {
				Z.dat <- finis
				break
			}
			for i := 1; i < n; i++ {
				<-Z.req
				Z.dat <- zero
			}
		}
	}(U, Z, c0, n)
	return Z
}

func Init() {
	chnameserial = -1
	seqno = 0
	chnames = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
	zero = itor(0)
	one = itor(1)
	finis = i2tor(1, 0)
	Ones = Rep(one)
	Twos = Rep(itor(2))
}

func check(U PS, c *rat, count int, str string) {
	for i := 0; i < count; i++ {
		r := get(U)
		if !r.eq(c) {
			print("got: ")
			r.pr()
			print("should get ")
			c.pr()
			print("\n")
			panic(str)
		}
	}
}

const N = 10

func checka(U PS, a []*rat, str string) {
	for i := 0; i < N; i++ {
		check(U, a[i], 1, str)
	}
}

func main() {
	Init()
	if len(os.Args) > 1 { // print
		print("Ones: ")
		Printn(Ones, 10)
		print("Twos: ")
		Printn(Twos, 10)
		print("Add: ")
		Printn(Add(Ones, Twos), 10)
		print("Diff: ")
		Printn(Diff(Ones), 10)
		print("Integ: ")
		Printn(Integ(zero, Ones), 10)
		print("CMul: ")
		Printn(Cmul(neg(one), Ones), 10)
		print("Sub: ")
		Printn(Sub(Ones, Twos), 10)
		print("Mul: ")
		Printn(Mul(Ones, Ones), 10)
		print("Exp: ")
		Printn(Exp(Ones), 15)
		print("MonSubst: ")
		Printn(MonSubst(Ones, neg(one), 2), 10)
		print("ATan: ")
		Printn(Integ(zero, MonSubst(Ones, neg(one), 2)), 10)
	} else { // test
		check(Ones, one, 5, "Ones")
		check(Add(Ones, Ones), itor(2), 0, "Add Ones Ones") // 1 1 1 1 1
		check(Add(Ones, Twos), itor(3), 0, "Add Ones Twos") // 3 3 3 3 3
		a := make([]*rat, N)
		d := Diff(Ones)
		for i := 0; i < N; i++ {
			a[i] = itor(int64(i + 1))
		}
		checka(d, a, "Diff") // 1 2 3 4 5
		in := Integ(zero, Ones)
		a[0] = zero // integration constant
		for i := 1; i < N; i++ {
			a[i] = i2tor(1, int64(i))
		}
		checka(in, a, "Integ")                               // 0 1 1/2 1/3 1/4 1/5
		check(Cmul(neg(one), Twos), itor(-2), 10, "CMul")    // -1 -1 -1 -1 -1
		check(Sub(Ones, Twos), itor(-1), 0, "Sub Ones Twos") // -1 -1 -1 -1 -1
		m := Mul(Ones, Ones)
		for i := 0; i < N; i++ {
			a[i] = itor(int64(i + 1))
		}
		checka(m, a, "Mul") // 1 2 3 4 5
		e := Exp(Ones)
		a[0] = itor(1)
		a[1] = itor(1)
		a[2] = i2tor(3, 2)
		a[3] = i2tor(13, 6)
		a[4] = i2tor(73, 24)
		a[5] = i2tor(167, 40)
		a[6] = i2tor(4051, 720)
		a[7] = i2tor(37633, 5040)
		a[8] = i2tor(43817, 4480)
		a[9] = i2tor(4596553, 362880)
		checka(e, a, "Exp") // 1 1 3/2 13/6 73/24
		at := Integ(zero, MonSubst(Ones, neg(one), 2))
		for c, i := 1, 0; i < N; i++ {
			if i%2 == 0 {
				a[i] = zero
			} else {
				a[i] = i2tor(int64(c), int64(i))
				c *= -1
			}
		}
		checka(at, a, "ATan") // 0 -1 0 -1/3 0 -1/5
		/*
			t := Revert(Integ(zero, MonSubst(Ones, neg(one), 2)))
			a[0] = zero
			a[1] = itor(1)
			a[2] = zero
			a[3] = i2tor(1,3)
			a[4] = zero
			a[5] = i2tor(2,15)
			a[6] = zero
			a[7] = i2tor(17,315)
			a[8] = zero
			a[9] = i2tor(62,2835)
			checka(t, a, "Tan")  // 0 1 0 1/3 0 2/15
		*/
	}
}
