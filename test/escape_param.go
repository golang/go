// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

// In this test almost everything is BAD except the simplest cases
// where input directly flows to output.

package escape

var sink interface{}

// in -> out
func param0(p *int) *int { // ERROR "leaking param: p to result ~r1"
	return p
}

func caller0a() {
	i := 0
	_ = param0(&i) // ERROR "caller0a &i does not escape$"
}

func caller0b() {
	i := 0            // ERROR "moved to heap: i$"
	sink = param0(&i) // ERROR "&i escapes to heap$" "param0\(&i\) escapes to heap"
}

// in, in -> out, out
func param1(p1, p2 *int) (*int, *int) { // ERROR "leaking param: p1 to result ~r2" "leaking param: p2 to result ~r3"
	return p1, p2
}

func caller1() {
	i := 0 // ERROR "moved to heap: i$"
	j := 0
	sink, _ = param1(&i, &j) // ERROR "&i escapes to heap$" "caller1 &j does not escape$"
}

// in -> other in
func param2(p1 *int, p2 **int) { // ERROR "leaking param: p1$" "param2 p2 does not escape$"
	*p2 = p1
}

func caller2a() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	param2(&i, &p) // ERROR "&i escapes to heap$" "caller2a &p does not escape$"
	_ = p
}

func caller2b() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	param2(&i, &p) // ERROR "&i escapes to heap$" "caller2b &p does not escape$"
	sink = p       // ERROR "p escapes to heap$"
}

// in -> in
type Pair struct {
	p1 *int
	p2 *int
}

func param3(p *Pair) { // ERROR "leaking param content: p$"
	p.p1 = p.p2
}

func caller3a() {
	i := 0            // ERROR "moved to heap: i$"
	j := 0            // ERROR "moved to heap: j$"
	p := Pair{&i, &j} // ERROR "&i escapes to heap$" "&j escapes to heap$"
	param3(&p)        // ERROR "caller3a &p does not escape"
	_ = p
}

func caller3b() {
	i := 0            // ERROR "moved to heap: i$"
	j := 0            // ERROR "moved to heap: j$"
	p := Pair{&i, &j} // ERROR "&i escapes to heap$" "&j escapes to heap$"
	param3(&p)        // ERROR "caller3b &p does not escape"
	sink = p          // ERROR "p escapes to heap$"
}

// in -> rcvr
func (p *Pair) param4(i *int) { // ERROR "\(\*Pair\).param4 p does not escape$" "leaking param: i$"
	p.p1 = i
}

func caller4a() {
	i := 0 // ERROR "moved to heap: i$"
	p := Pair{}
	p.param4(&i) // ERROR "&i escapes to heap$" "caller4a p does not escape$"
	_ = p
}

func caller4b() {
	i := 0 // ERROR "moved to heap: i$"
	p := Pair{}
	p.param4(&i) // ERROR "&i escapes to heap$" "caller4b p does not escape$"
	sink = p     // ERROR "p escapes to heap$"
}

// in -> heap
func param5(i *int) { // ERROR "leaking param: i$"
	sink = i // ERROR "i escapes to heap$"
}

func caller5() {
	i := 0     // ERROR "moved to heap: i$"
	param5(&i) // ERROR "&i escapes to heap$"
}

// *in -> heap
func param6(i ***int) { // ERROR "leaking param content: i$"
	sink = *i // ERROR "\*i escapes to heap$"
}

func caller6a() {
	i := 0      // ERROR "moved to heap: i$"
	p := &i     // ERROR "&i escapes to heap$" "moved to heap: p$"
	p2 := &p    // ERROR "&p escapes to heap$"
	param6(&p2) // ERROR "caller6a &p2 does not escape"
}

// **in -> heap
func param7(i ***int) { // ERROR "leaking param content: i$"
	sink = **i // ERROR "\* \(\*i\) escapes to heap"
}

func caller7() {
	i := 0      // ERROR "moved to heap: i$"
	p := &i     // ERROR "&i escapes to heap$" "moved to heap: p$"
	p2 := &p    // ERROR "&p escapes to heap$"
	param7(&p2) // ERROR "caller7 &p2 does not escape"
}

// **in -> heap
func param8(i **int) { // ERROR "param8 i does not escape$"
	sink = **i // ERROR "\* \(\*i\) escapes to heap"
}

func caller8() {
	i := 0
	p := &i    // ERROR "caller8 &i does not escape$"
	param8(&p) // ERROR "caller8 &p does not escape$"
}

// *in -> out
func param9(p ***int) **int { // ERROR "leaking param: p to result ~r1 level=1"
	return *p
}

func caller9a() {
	i := 0
	p := &i         // ERROR "caller9a &i does not escape"
	p2 := &p        // ERROR "caller9a &p does not escape"
	_ = param9(&p2) // ERROR "caller9a &p2 does not escape$"
}

func caller9b() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "&i escapes to heap$" "moved to heap: p$"
	p2 := &p           // ERROR "&p escapes to heap$"
	sink = param9(&p2) // ERROR "caller9b &p2 does not escape$"  "param9\(&p2\) escapes to heap"
}

// **in -> out
func param10(p ***int) *int { // ERROR "leaking param: p to result ~r1 level=2"
	return **p
}

func caller10a() {
	i := 0
	p := &i          // ERROR "caller10a &i does not escape"
	p2 := &p         // ERROR "caller10a &p does not escape"
	_ = param10(&p2) // ERROR "caller10a &p2 does not escape$"
}

func caller10b() {
	i := 0              // ERROR "moved to heap: i$"
	p := &i             // ERROR "&i escapes to heap$"
	p2 := &p            // ERROR "caller10b &p does not escape$"
	sink = param10(&p2) // ERROR "caller10b &p2 does not escape$" "param10\(&p2\) escapes to heap"
}

// in escapes to heap (address of param taken and returned)
func param11(i **int) ***int { // ERROR "moved to heap: i$"
	return &i // ERROR "&i escapes to heap$"
}

func caller11a() {
	i := 0          // ERROR "moved to heap: i"
	p := &i         // ERROR "moved to heap: p" "&i escapes to heap"
	_ = param11(&p) // ERROR "&p escapes to heap"
}

func caller11b() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "&i escapes to heap$" "moved to heap: p$"
	sink = param11(&p) // ERROR "&p escapes to heap$" "param11\(&p\) escapes to heap"
}

func caller11c() { // GOOD
	i := 0              // ERROR "moved to heap: i$"
	p := &i             // ERROR "moved to heap: p" "&i escapes to heap"
	sink = *param11(&p) // ERROR "&p escapes to heap" "\*param11\(&p\) escapes to heap"
}

func caller11d() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "&i escapes to heap" "moved to heap: p"
	p2 := &p           // ERROR "&p escapes to heap"
	sink = param11(p2) // ERROR "param11\(p2\) escapes to heap"
}

// &in -> rcvr
type Indir struct {
	p ***int
}

func (r *Indir) param12(i **int) { // ERROR "\(\*Indir\).param12 r does not escape$" "moved to heap: i$"
	r.p = &i // ERROR "&i escapes to heap$"
}

func caller12a() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "&i escapes to heap$" "moved to heap: p$"
	var r Indir
	r.param12(&p) // ERROR "&p escapes to heap$" "caller12a r does not escape$"
	_ = r
}

func caller12b() {
	i := 0        // ERROR "moved to heap: i$"
	p := &i       // ERROR "&i escapes to heap$" "moved to heap: p$"
	r := &Indir{} // ERROR "caller12b &Indir literal does not escape$"
	r.param12(&p) // ERROR "&p escapes to heap$"
	_ = r
}

func caller12c() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "&i escapes to heap$" "moved to heap: p$"
	r := Indir{}
	r.param12(&p) // ERROR "&p escapes to heap$" "caller12c r does not escape$"
	sink = r      // ERROR "r escapes to heap$"
}

func caller12d() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "&i escapes to heap$" "moved to heap: p$"
	r := Indir{}
	r.param12(&p) // ERROR "&p escapes to heap$" "caller12d r does not escape$"
	sink = **r.p  // ERROR "\* \(\*r\.p\) escapes to heap"
}

// in -> value rcvr
type Val struct {
	p **int
}

func (v Val) param13(i *int) { // ERROR "Val.param13 v does not escape$" "leaking param: i$"
	*v.p = i
}

func caller13a() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	var v Val
	v.p = &p      // ERROR "caller13a &p does not escape$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	_ = v
}

func caller13b() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := Val{&p}  // ERROR "caller13b &p does not escape$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	_ = v
}

func caller13c() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := &Val{&p} // ERROR "caller13c &Val literal does not escape$" "caller13c &p does not escape$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	_ = v
}

func caller13d() {
	i := 0     // ERROR "moved to heap: i$"
	var p *int // ERROR "moved to heap: p$"
	var v Val
	v.p = &p      // ERROR "&p escapes to heap$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	sink = v      // ERROR "v escapes to heap$"
}

func caller13e() {
	i := 0        // ERROR "moved to heap: i$"
	var p *int    // ERROR "moved to heap: p$"
	v := Val{&p}  // ERROR "&p escapes to heap$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	sink = v      // ERROR "v escapes to heap$"
}

func caller13f() {
	i := 0        // ERROR "moved to heap: i$"
	var p *int    // ERROR "moved to heap: p$"
	v := &Val{&p} // ERROR "&Val literal escapes to heap$" "&p escapes to heap$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	sink = v      // ERROR "v escapes to heap$"
}

func caller13g() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := Val{&p}  // ERROR "caller13g &p does not escape$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	sink = *v.p   // ERROR "\*v\.p escapes to heap"
}

func caller13h() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := &Val{&p} // ERROR "caller13h &Val literal does not escape$" "caller13h &p does not escape$"
	v.param13(&i) // ERROR "&i escapes to heap$"
	sink = **v.p  // ERROR "\* \(\*v\.p\) escapes to heap"
}

type Node struct {
	p *Node
}

var Sink *Node

func f(x *Node) { // ERROR "leaking param content: x"
	Sink = &Node{x.p} // ERROR "&Node literal escapes to heap"
}

func g(x *Node) *Node { // ERROR "leaking param: x to result ~r1 level=0"
	return &Node{x.p} // ERROR "&Node literal escapes to heap"
}

func h(x *Node) { // ERROR "leaking param: x"
	y := &Node{x} // ERROR "h &Node literal does not escape"
	Sink = g(y)
	f(y)
}
