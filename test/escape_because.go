// errorcheck -0 -m -m -l -newescape=false

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note the doubled -m; this tests the "because" explanations for escapes,
// and is likely to be annoyingly fragile under compiler change.
// As long as the explanations look reasonably sane, meaning eyeball verify output of
//    go build -gcflags '-l -m -m' escape_because.go
// and investigate changes, feel free to update with
//    go run run.go -update_errors -- escape_because.go

package main

func main() {
}

var sink interface{}

type pair struct {
	x, y *int
}

type Pairy interface {
	EqualParts() bool
}

func (p *pair) EqualParts() bool { // ERROR "\(\*pair\).EqualParts p does not escape$"
	return p != nil && (p.x == p.y || *p.x == *p.y)
}

func f1(p *int) { // ERROR "from \[3\]\*int literal \(array literal element\) at escape_because.go:34$" "from a \(assigned\) at escape_because.go:34$" "from a \(interface-converted\) at escape_because.go:35$" "from sink \(assigned to top level variable\) at escape_because.go:35$" "leaking param: p$"
	a := [3]*int{p, nil, nil}
	sink = a // ERROR "a escapes to heap$" "from sink \(assigned to top level variable\) at escape_because.go:35$"

}

func f2(q *int) { // ERROR "from &u \(address-of\) at escape_because.go:43$" "from &u \(interface-converted\) at escape_because.go:43$" "from pair literal \(struct literal element\) at escape_because.go:41$" "from s \(assigned\) at escape_because.go:40$" "from sink \(assigned to top level variable\) at escape_because.go:43$" "from t \(assigned\) at escape_because.go:41$" "from u \(assigned\) at escape_because.go:42$" "leaking param: q$"
	s := q
	t := pair{s, nil}
	u := t    // ERROR "moved to heap: u$"
	sink = &u // ERROR "&u escapes to heap$" "from sink \(assigned to top level variable\) at escape_because.go:43$"
}

func f3(r *int) interface{} { // ERROR "from \[\]\*int literal \(slice-literal-element\) at escape_because.go:47$" "from c \(assigned\) at escape_because.go:47$" "from c \(interface-converted\) at escape_because.go:48$" "from ~r1 \(return\) at escape_because.go:48$" "leaking param: r"
	c := []*int{r} // ERROR "\[\]\*int literal escapes to heap$" "from c \(assigned\) at escape_because.go:47$" "from c \(interface-converted\) at escape_because.go:48$" "from ~r1 \(return\) at escape_because.go:48$"
	return c       // "return" // ERROR "c escapes to heap$" "from ~r1 \(return\) at escape_because.go:48$"
}

func f4(a *int, s []*int) int { // ERROR "from \*s \(indirection\) at escape_because.go:51$" "from append\(s, a\) \(appended to slice\) at escape_because.go:52$" "from append\(s, a\) \(appendee slice\) at escape_because.go:52$" "leaking param content: s$" "leaking param: a$"
	s = append(s, a)
	return *(s[0])
}

func f5(s1, s2 []*int) int { // ERROR "from \*s1 \(indirection\) at escape_because.go:56$" "from \*s2 \(indirection\) at escape_because.go:56$" "from append\(s1, s2...\) \(appended slice...\) at escape_because.go:57$" "from append\(s1, s2...\) \(appendee slice\) at escape_because.go:57$" "leaking param content: s1$" "leaking param content: s2$"
	s1 = append(s1, s2...)
	return *(s1[0])
}

func f6(x, y *int) bool { // ERROR "f6 x does not escape$" "f6 y does not escape$"
	p := pair{x, y}
	var P Pairy = &p // ERROR "f6 &p does not escape$"
	pp := P.(*pair)
	return pp.EqualParts()
}

func f7(x map[int]*int, y int) *int { // ERROR "f7 x does not escape$"
	z, ok := x[y]
	if !ok {
		return nil
	}
	return z
}

func f8(x int, y *int) *int { // ERROR "from ~r2 \(return\) at escape_because.go:78$" "from ~r2 \(returned from recursive function\) at escape_because.go:76$" "leaking param: y$" "moved to heap: x$"
	if x <= 0 {
		return y
	}
	x--
	return f8(*y, &x)
}

func f9(x int, y ...*int) *int { // ERROR "from y\[0\] \(dot of pointer\) at escape_because.go:86$" "from ~r2 \(return\) at escape_because.go:86$" "from ~r2 \(returned from recursive function\) at escape_because.go:84$" "leaking param content: y$" "leaking param: y to result ~r2 level=1$" "moved to heap: x$"
	if x <= 0 {
		return y[0]
	}
	x--
	return f9(*y[0], &x) // ERROR "f9 ... argument does not escape$"
}

func f10(x map[*int]*int, y, z *int) *int { // ERROR "f10 x does not escape$" "from x\[y\] \(key of map put\) at escape_because.go:93$" "from x\[y\] \(value of map put\) at escape_because.go:93$" "leaking param: y$" "leaking param: z$"
	x[y] = z
	return z
}

func f11(x map[*int]*int, y, z *int) map[*int]*int { // ERROR "f11 x does not escape$" "from map\[\*int\]\*int literal \(map literal key\) at escape_because.go:98$" "from map\[\*int\]\*int literal \(map literal value\) at escape_because.go:98$" "leaking param: y$" "leaking param: z$"
	return map[*int]*int{y: z} // ERROR "from ~r3 \(return\) at escape_because.go:98$" "map\[\*int\]\*int literal escapes to heap$"
}

func f12() {
	b := []byte("test") // ERROR "\(\[\]byte\)\(.test.\) escapes to heap$" "from b \(assigned\) at escape_because.go:102$" "from b \(passed to call\[argument escapes\]\) at escape_because.go:103$"
	escape(b)
}

func escape(b []byte) { // ERROR "from panic\(b\) \(panic\) at escape_because.go:107$" "leaking param: b$"
	panic(b)
}

func f13() {
	b := []byte("test") // ERROR "\(\[\]byte\)\(.test.\) escapes to heap$" "from .out0 \(passed-to-and-returned-from-call\) at escape_because.go:112$" "from b \(assigned\) at escape_because.go:111$" "from c \(assigned\) at escape_because.go:112$" "from c \(passed to call\[argument escapes\]\) at escape_because.go:113$"
	c := transmit(b)
	escape(c)
}

func transmit(b []byte) []byte { // ERROR "from ~r1 \(return\) at escape_because.go:117$" "leaking param: b to result ~r1 level=0$"
	return b
}

func f14() {
	n := 32
	s1 := make([]int, n)    // ERROR "make\(\[\]int, n\) escapes to heap" "from make\(\[\]int, n\) \(non-constant size\)"
	s2 := make([]int, 0, n) // ERROR "make\(\[\]int, 0, n\) escapes to heap" "from make\(\[\]int, 0, n\) \(non-constant size\)"
	_, _ = s1, s2
}

func leakParams(p1, p2 *int) (*int, *int) { // ERROR "leaking param: p1 to result ~r2 level=0$" "from ~r2 \(return\) at escape_because.go:128$" "leaking param: p2 to result ~r3 level=0$" "from ~r3 \(return\) at escape_because.go:128$"
	return p1, p2
}

func leakThroughOAS2() {
	// See #26987.
	i := 0              // ERROR "moved to heap: i$"
	j := 0              // ERROR "moved to heap: j$"
	sink, sink = &i, &j // ERROR "&i escapes to heap$" "from sink \(assign-pair\) at escape_because.go:135$" "&j escapes to heap$"
}

func leakThroughOAS2FUNC() {
	// See #26987.
	i := 0 // ERROR "moved to heap: i$"
	j := 0
	sink, _ = leakParams(&i, &j)
}

// The list below is all of the why-escapes messages seen building the escape analysis tests.
/*
   for i in escape*go ; do echo compile $i; go build -gcflags '-l -m -m' $i >& `basename $i .go`.log ; done
   grep 'from .* at ' escape*.log | sed -e 's/^.*(\([^()]*\))[^()]*$/\1/' | sort -u
*/
// sed RE above assumes that (reason) is the last parenthesized phrase in the line,
// and that none of the reasons contains any parentheses

/*
... arg to recursive call
address-of
appended slice...
appended to slice
appendee slice
arg to ...
arg to recursive call
array-element-equals
array literal element
assigned
assigned to top level variable
assign-pair-dot-type
assign-pair-func-call
captured by a closure
captured by called closure
dot
dot-equals
dot of pointer
fixed-array-index-of
go func arg
indirection
interface-converted
key of map put
map literal key
map literal value
non-constant size
panic
parameter to indirect call
passed-to-and-returned-from-call
passed to call[argument content escapes]
passed to call[argument escapes]
pointer literal
range-deref
receiver in indirect call
return
returned from recursive function
slice-element-equals
slice-literal-element
star-dot-equals
star-equals
struct literal element
too large for stack
value of map put
*/

// Expected, but not yet seen (they may be unreachable):

/*
append-first-arg
assign-pair-mapr
assign-pair-receive
call receiver
map index
pointer literal [assign]
slice literal element
*/
