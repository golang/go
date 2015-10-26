// +build ignore

package main

// Test of maps.

var a, b, c int

func maps1() {
	m1 := map[*int]*int{&a: &b} // @line m1m1
	m2 := make(map[*int]*int)   // @line m1m2
	m2[&b] = &a

	print(m1[nil]) // @pointsto main.b | main.c
	print(m2[nil]) // @pointsto main.a

	print(m1) // @pointsto makemap@m1m1:21
	print(m2) // @pointsto makemap@m1m2:12

	m1[&b] = &c

	for k, v := range m1 {
		print(k) // @pointsto main.a | main.b
		print(v) // @pointsto main.b | main.c
	}

	for k, v := range m2 {
		print(k) // @pointsto main.b
		print(v) // @pointsto main.a
	}

	// Lookup doesn't create any aliases.
	print(m2[&c]) // @pointsto main.a
	if _, ok := m2[&a]; ok {
		print(m2[&c]) // @pointsto main.a
	}
}

func maps2() {
	m1 := map[*int]*int{&a: &b}
	m2 := map[*int]*int{&b: &c}
	_ = []map[*int]*int{m1, m2} // (no spurious merging of m1, m2)

	print(m1[nil]) // @pointsto main.b
	print(m2[nil]) // @pointsto main.c
}

var g int

func maps3() {
	// Regression test for a constraint generation bug for map range
	// loops in which the key is unused: the (ok, k, v) tuple
	// returned by ssa.Next may have type 'invalid' for the k and/or
	// v components, so copying the map key or value may cause
	// miswiring if the key has >1 components.  In the worst case,
	// this causes a crash.  The test below used to report that
	// pts(v) includes not just main.g but new(float64) too, which
	// is ill-typed.

	// sizeof(K) > 1, abstractly
	type K struct{ a, b *float64 }
	k := K{new(float64), nil}
	m := map[K]*int{k: &g}

	for _, v := range m {
		print(v) // @pointsto main.g
	}
}

func main() {
	maps1()
	maps2()
	maps3()
}
