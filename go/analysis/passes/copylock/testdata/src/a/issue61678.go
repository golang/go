package a

import "sync"

// These examples are taken from golang/go#61678, modified so that A and B
// contain a mutex.

type A struct {
	a  A
	mu sync.Mutex
}

type B struct {
	a  A
	b  B
	mu sync.Mutex
}

func okay(x A) {}
func sure()    { var x A; nop(x) }

var fine B

func what(x B)   {}                  // want `passes lock by value`
func bad()       { var x B; nop(x) } // want `copies lock value`
func good()      { nop(B{}) }
func stillgood() { nop(B{b: B{b: B{b: B{}}}}) }
func nope()      { nop(B{}.b) } // want `copies lock value`

func nop(any) {} // only used to get around unused variable errors
