// errorcheck -0 -m -l -newescape=true

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

// In this test almost everything is BAD except the simplest cases
// where input directly flows to output.

package escape

func zero() int { return 0 }

var sink interface{}

// in -> out
func param0(p *int) *int { // ERROR "leaking param: p to result ~r1"
	return p
}

func caller0a() {
	i := 0
	_ = param0(&i)
}

func caller0b() {
	i := 0            // ERROR "moved to heap: i$"
	sink = param0(&i) // ERROR "param0\(&i\) escapes to heap"
}

// in, in -> out, out
func param1(p1, p2 *int) (*int, *int) { // ERROR "leaking param: p1 to result ~r2" "leaking param: p2 to result ~r3"
	return p1, p2
}

func caller1() {
	i := 0 // ERROR "moved to heap: i$"
	j := 0
	sink, _ = param1(&i, &j)
}

// in -> other in
func param2(p1 *int, p2 **int) { // ERROR "leaking param: p1$" "param2 p2 does not escape$"
	*p2 = p1
}

func caller2a() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	param2(&i, &p)
	_ = p
}

func caller2b() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	param2(&i, &p)
	sink = p       // ERROR "p escapes to heap$"
}

func paramArraySelfAssign(p *PairOfPairs) { // ERROR "p does not escape"
	p.pairs[0] = p.pairs[1] // ERROR "ignoring self-assignment in p.pairs\[0\] = p.pairs\[1\]"
}

func paramArraySelfAssignUnsafeIndex(p *PairOfPairs) { // ERROR "leaking param content: p"
	// Function call inside index disables self-assignment case to trigger.
	p.pairs[zero()] = p.pairs[1]
	p.pairs[zero()+1] = p.pairs[1]
}

type PairOfPairs struct {
	pairs [2]*Pair
}

type BoxedPair struct {
	pair *Pair
}

type WrappedPair struct {
	pair Pair
}

func leakParam(x interface{}) { // ERROR "leaking param: x"
	sink = x
}

func sinkAfterSelfAssignment1(box *BoxedPair) { // ERROR "leaking param content: box"
	box.pair.p1 = box.pair.p2 // ERROR "ignoring self-assignment in box.pair.p1 = box.pair.p2"
	sink = box.pair.p2        // ERROR "box.pair.p2 escapes to heap"
}

func sinkAfterSelfAssignment2(box *BoxedPair) { // ERROR "leaking param content: box"
	box.pair.p1 = box.pair.p2 // ERROR "ignoring self-assignment in box.pair.p1 = box.pair.p2"
	sink = box.pair           // ERROR "box.pair escapes to heap"
}

func sinkAfterSelfAssignment3(box *BoxedPair) { // ERROR "leaking param content: box"
	box.pair.p1 = box.pair.p2 // ERROR "ignoring self-assignment in box.pair.p1 = box.pair.p2"
	leakParam(box.pair.p2)    // ERROR "box.pair.p2 escapes to heap"
}

func sinkAfterSelfAssignment4(box *BoxedPair) { // ERROR "leaking param content: box"
	box.pair.p1 = box.pair.p2 // ERROR "ignoring self-assignment in box.pair.p1 = box.pair.p2"
	leakParam(box.pair)       // ERROR "box.pair escapes to heap"
}

func selfAssignmentAndUnrelated(box1, box2 *BoxedPair) { // ERROR "leaking param content: box2" "box1 does not escape"
	box1.pair.p1 = box1.pair.p2 // ERROR "ignoring self-assignment in box1.pair.p1 = box1.pair.p2"
	leakParam(box2.pair.p2)     // ERROR "box2.pair.p2 escapes to heap"
}

func notSelfAssignment1(box1, box2 *BoxedPair) { // ERROR "leaking param content: box2" "box1 does not escape"
	box1.pair.p1 = box2.pair.p1
}

func notSelfAssignment2(p1, p2 *PairOfPairs) { // ERROR "leaking param content: p2" "p1 does not escape"
	p1.pairs[0] = p2.pairs[1]
}

func notSelfAssignment3(p1, p2 *PairOfPairs) { // ERROR "leaking param content: p2" "p1 does not escape"
	p1.pairs[0].p1 = p2.pairs[1].p1
}

func boxedPairSelfAssign(box *BoxedPair) { // ERROR "box does not escape"
	box.pair.p1 = box.pair.p2 // ERROR "ignoring self-assignment in box.pair.p1 = box.pair.p2"
}

func wrappedPairSelfAssign(w *WrappedPair) { // ERROR "w does not escape"
	w.pair.p1 = w.pair.p2 // ERROR "ignoring self-assignment in w.pair.p1 = w.pair.p2"
}

// in -> in
type Pair struct {
	p1 *int
	p2 *int
}

func param3(p *Pair) { // ERROR "param3 p does not escape"
	p.p1 = p.p2 // ERROR "param3 ignoring self-assignment in p.p1 = p.p2"
}

func caller3a() {
	i := 0
	j := 0
	p := Pair{&i, &j}
	param3(&p)
	_ = p
}

func caller3b() {
	i := 0            // ERROR "moved to heap: i$"
	j := 0            // ERROR "moved to heap: j$"
	p := Pair{&i, &j}
	param3(&p)
	sink = p          // ERROR "p escapes to heap$"
}

// in -> rcvr
func (p *Pair) param4(i *int) { // ERROR "\(\*Pair\).param4 p does not escape$" "leaking param: i$"
	p.p1 = i
}

func caller4a() {
	i := 0 // ERROR "moved to heap: i$"
	p := Pair{}
	p.param4(&i)
	_ = p
}

func caller4b() {
	i := 0 // ERROR "moved to heap: i$"
	p := Pair{}
	p.param4(&i)
	sink = p     // ERROR "p escapes to heap$"
}

// in -> heap
func param5(i *int) { // ERROR "leaking param: i$"
	sink = i // ERROR "i escapes to heap$"
}

func caller5() {
	i := 0     // ERROR "moved to heap: i$"
	param5(&i)
}

// *in -> heap
func param6(i ***int) { // ERROR "leaking param content: i$"
	sink = *i // ERROR "\*i escapes to heap$"
}

func caller6a() {
	i := 0      // ERROR "moved to heap: i$"
	p := &i     // ERROR "moved to heap: p$"
	p2 := &p
	param6(&p2)
}

// **in -> heap
func param7(i ***int) { // ERROR "leaking param content: i$"
	sink = **i // ERROR "\* \(\*i\) escapes to heap"
}

func caller7() {
	i := 0      // ERROR "moved to heap: i$"
	p := &i     // ERROR "moved to heap: p$"
	p2 := &p
	param7(&p2)
}

// **in -> heap
func param8(i **int) { // ERROR "param8 i does not escape$"
	sink = **i // ERROR "\* \(\*i\) escapes to heap"
}

func caller8() {
	i := 0
	p := &i
	param8(&p)
}

// *in -> out
func param9(p ***int) **int { // ERROR "leaking param: p to result ~r1 level=1"
	return *p
}

func caller9a() {
	i := 0
	p := &i
	p2 := &p
	_ = param9(&p2)
}

func caller9b() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "moved to heap: p$"
	p2 := &p
	sink = param9(&p2) // ERROR  "param9\(&p2\) escapes to heap"
}

// **in -> out
func param10(p ***int) *int { // ERROR "leaking param: p to result ~r1 level=2"
	return **p
}

func caller10a() {
	i := 0
	p := &i
	p2 := &p
	_ = param10(&p2)
}

func caller10b() {
	i := 0              // ERROR "moved to heap: i$"
	p := &i
	p2 := &p
	sink = param10(&p2) // ERROR "param10\(&p2\) escapes to heap"
}

// in escapes to heap (address of param taken and returned)
func param11(i **int) ***int { // ERROR "moved to heap: i$"
	return &i
}

func caller11a() {
	i := 0          // ERROR "moved to heap: i"
	p := &i         // ERROR "moved to heap: p"
	_ = param11(&p)
}

func caller11b() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "moved to heap: p$"
	sink = param11(&p) // ERROR "param11\(&p\) escapes to heap"
}

func caller11c() { // GOOD
	i := 0              // ERROR "moved to heap: i$"
	p := &i             // ERROR "moved to heap: p"
	sink = *param11(&p) // ERROR "\*param11\(&p\) escapes to heap"
}

func caller11d() {
	i := 0             // ERROR "moved to heap: i$"
	p := &i            // ERROR "moved to heap: p"
	p2 := &p
	sink = param11(p2) // ERROR "param11\(p2\) escapes to heap"
}

// &in -> rcvr
type Indir struct {
	p ***int
}

func (r *Indir) param12(i **int) { // ERROR "\(\*Indir\).param12 r does not escape$" "moved to heap: i$"
	r.p = &i
}

func caller12a() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "moved to heap: p$"
	var r Indir
	r.param12(&p)
	_ = r
}

func caller12b() {
	i := 0        // ERROR "moved to heap: i$"
	p := &i       // ERROR "moved to heap: p$"
	r := &Indir{} // ERROR "caller12b &Indir literal does not escape$"
	r.param12(&p)
	_ = r
}

func caller12c() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "moved to heap: p$"
	r := Indir{}
	r.param12(&p)
	sink = r      // ERROR "r escapes to heap$"
}

func caller12d() {
	i := 0  // ERROR "moved to heap: i$"
	p := &i // ERROR "moved to heap: p$"
	r := Indir{}
	r.param12(&p)
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
	v.p = &p
	v.param13(&i)
	_ = v
}

func caller13b() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := Val{&p}
	v.param13(&i)
	_ = v
}

func caller13c() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := &Val{&p} // ERROR "caller13c &Val literal does not escape$"
	v.param13(&i)
	_ = v
}

func caller13d() {
	i := 0     // ERROR "moved to heap: i$"
	var p *int // ERROR "moved to heap: p$"
	var v Val
	v.p = &p
	v.param13(&i)
	sink = v      // ERROR "v escapes to heap$"
}

func caller13e() {
	i := 0        // ERROR "moved to heap: i$"
	var p *int    // ERROR "moved to heap: p$"
	v := Val{&p}
	v.param13(&i)
	sink = v      // ERROR "v escapes to heap$"
}

func caller13f() {
	i := 0        // ERROR "moved to heap: i$"
	var p *int    // ERROR "moved to heap: p$"
	v := &Val{&p} // ERROR "&Val literal escapes to heap$"
	v.param13(&i)
	sink = v      // ERROR "v escapes to heap$"
}

func caller13g() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := Val{&p}
	v.param13(&i)
	sink = *v.p   // ERROR "\*v\.p escapes to heap"
}

func caller13h() {
	i := 0 // ERROR "moved to heap: i$"
	var p *int
	v := &Val{&p} // ERROR "caller13h &Val literal does not escape$"
	v.param13(&i)
	sink = **v.p  // ERROR "\* \(\*v\.p\) escapes to heap"
}

type Node struct {
	p *Node
}

var Sink *Node

func f(x *Node) { // ERROR "leaking param content: x"
	Sink = &Node{x.p} // ERROR "&Node literal escapes to heap"
}

func g(x *Node) *Node { // ERROR "leaking param content: x"
	return &Node{x.p} // ERROR "&Node literal escapes to heap"
}

func h(x *Node) { // ERROR "leaking param: x"
	y := &Node{x} // ERROR "h &Node literal does not escape"
	Sink = g(y)
	f(y)
}

// interface(in) -> out
// See also issue 29353.

// Convert to a non-direct interface, require an allocation and
// copy x to heap (not to result).
func param14a(x [4]*int) interface{} { // ERROR "leaking param: x$"
	return x // ERROR "x escapes to heap"
}

// Convert to a direct interface, does not need an allocation.
// So x only leaks to result.
func param14b(x *int) interface{} { // ERROR "leaking param: x to result ~r1 level=0"
	return x // ERROR "x escapes to heap"
}
