// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

// Constraint type inference should be independent of the
// ordering of the type parameter declarations. Try all
// permutations in the test case below.
// Permutations produced by https://go.dev/play/p/PHcZNGJTEBZ.

func f00[S1 ~[]E1, S2 ~[]E2, E1 ~byte, E2 ~byte](S1, S2) {}
func f01[S2 ~[]E2, S1 ~[]E1, E1 ~byte, E2 ~byte](S1, S2) {}
func f02[E1 ~byte, S1 ~[]E1, S2 ~[]E2, E2 ~byte](S1, S2) {}
func f03[S1 ~[]E1, E1 ~byte, S2 ~[]E2, E2 ~byte](S1, S2) {}
func f04[S2 ~[]E2, E1 ~byte, S1 ~[]E1, E2 ~byte](S1, S2) {}
func f05[E1 ~byte, S2 ~[]E2, S1 ~[]E1, E2 ~byte](S1, S2) {}
func f06[E2 ~byte, S2 ~[]E2, S1 ~[]E1, E1 ~byte](S1, S2) {}
func f07[S2 ~[]E2, E2 ~byte, S1 ~[]E1, E1 ~byte](S1, S2) {}
func f08[S1 ~[]E1, E2 ~byte, S2 ~[]E2, E1 ~byte](S1, S2) {}
func f09[E2 ~byte, S1 ~[]E1, S2 ~[]E2, E1 ~byte](S1, S2) {}
func f10[S2 ~[]E2, S1 ~[]E1, E2 ~byte, E1 ~byte](S1, S2) {}
func f11[S1 ~[]E1, S2 ~[]E2, E2 ~byte, E1 ~byte](S1, S2) {}
func f12[S1 ~[]E1, E1 ~byte, E2 ~byte, S2 ~[]E2](S1, S2) {}
func f13[E1 ~byte, S1 ~[]E1, E2 ~byte, S2 ~[]E2](S1, S2) {}
func f14[E2 ~byte, S1 ~[]E1, E1 ~byte, S2 ~[]E2](S1, S2) {}
func f15[S1 ~[]E1, E2 ~byte, E1 ~byte, S2 ~[]E2](S1, S2) {}
func f16[E1 ~byte, E2 ~byte, S1 ~[]E1, S2 ~[]E2](S1, S2) {}
func f17[E2 ~byte, E1 ~byte, S1 ~[]E1, S2 ~[]E2](S1, S2) {}
func f18[E2 ~byte, E1 ~byte, S2 ~[]E2, S1 ~[]E1](S1, S2) {}
func f19[E1 ~byte, E2 ~byte, S2 ~[]E2, S1 ~[]E1](S1, S2) {}
func f20[S2 ~[]E2, E2 ~byte, E1 ~byte, S1 ~[]E1](S1, S2) {}
func f21[E2 ~byte, S2 ~[]E2, E1 ~byte, S1 ~[]E1](S1, S2) {}
func f22[E1 ~byte, S2 ~[]E2, E2 ~byte, S1 ~[]E1](S1, S2) {}
func f23[S2 ~[]E2, E1 ~byte, E2 ~byte, S1 ~[]E1](S1, S2) {}

type myByte byte

func _(a []byte, b []myByte) {
	f00(a, b)
	f01(a, b)
	f02(a, b)
	f03(a, b)
	f04(a, b)
	f05(a, b)
	f06(a, b)
	f07(a, b)
	f08(a, b)
	f09(a, b)
	f10(a, b)
	f11(a, b)
	f12(a, b)
	f13(a, b)
	f14(a, b)
	f15(a, b)
	f16(a, b)
	f17(a, b)
	f18(a, b)
	f19(a, b)
	f20(a, b)
	f21(a, b)
	f22(a, b)
	f23(a, b)
}

// Constraint type inference may have to iterate.
// Again, the order of the type parameters shouldn't matter.

func g0[S ~[]E, M ~map[string]S, E any](m M) {}
func g1[M ~map[string]S, S ~[]E, E any](m M) {}
func g2[E any, S ~[]E, M ~map[string]S](m M) {}
func g3[S ~[]E, E any, M ~map[string]S](m M) {}
func g4[M ~map[string]S, E any, S ~[]E](m M) {}
func g5[E any, M ~map[string]S, S ~[]E](m M) {}

func _(m map[string][]byte) {
	g0(m)
	g1(m)
	g2(m)
	g3(m)
	g4(m)
	g5(m)
}

// Worst-case scenario.
// There are 10 unknown type parameters. In each iteration of
// constraint type inference we infer one more, from right to left.
// Each iteration looks repeatedly at all 11 type parameters,
// requiring a total of 10*11 = 110 iterations with the current
// implementation. Pathological case.

func h[K any, J ~*K, I ~*J, H ~*I, G ~*H, F ~*G, E ~*F, D ~*E, C ~*D, B ~*C, A ~*B](x A) {}

func _(x **********int) {
	h(x)
}

// Examples with channel constraints and tilde.

func ch1[P chan<- int]() (_ P)           { return } // core(P) == chan<- int   (single type, no tilde)
func ch2[P ~chan int]()                  { return } // core(P) == ~chan<- int  (tilde)
func ch3[P chan E, E any](E)             { return } // core(P) == chan<- E     (single type, no tilde)
func ch4[P chan E | ~chan<- E, E any](E) { return } // core(P) == ~chan<- E    (tilde)
func ch5[P chan int | chan<- int]()      { return } // core(P) == chan<- int   (not a single type)

func _() {
	// P can be inferred as there's a single specific type and no tilde.
	var _ chan int = ch1 /* ERRORx `cannot use ch1.*value of type chan<- int` */ ()
	var _ chan<- int = ch1()

	// P cannot be inferred as there's a tilde.
	ch2 /* ERROR "cannot infer P" */ ()
	type myChan chan int
	ch2[myChan]()

	// P can be inferred as there's a single specific type and no tilde.
	var e int
	ch3(e)

	// P cannot be inferred as there's more than one specific type and a tilde.
	ch4 /* ERROR "cannot infer P" */ (e)
	_ = ch4[chan int]

	// P cannot be inferred as there's more than one specific type.
	ch5 /* ERROR "cannot infer P" */ ()
	ch5[chan<- int]()
}

// test case from issue

func equal[M1 ~map[K1]V1, M2 ~map[K2]V2, K1, K2 ~uint32, V1, V2 ~string](m1 M1, m2 M2) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[K2(k)]; !ok || V2(v1) != v2 {
			return false
		}
	}
	return true
}

func equalFixed[K1, K2 ~uint32, V1, V2 ~string](m1 map[K1]V1, m2 map[K2]V2) bool {
	if len(m1) != len(m2) {
		return false
	}
	for k, v1 := range m1 {
		if v2, ok := m2[K2(k)]; !ok || v1 != V1(v2) {
			return false
		}
	}
	return true
}

type (
	someNumericID uint32
	someStringID  string
)

func _() {
	foo := map[uint32]string{10: "bar"}
	bar := map[someNumericID]someStringID{10: "bar"}
	equal(foo, bar)
}
