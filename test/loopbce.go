// +build amd64
// errorcheck -0 -d=ssa/loopbce/debug=3

package main

func f0a(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable with minimum 0 and increment 1$"
		x += a[i] // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func f0b(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable with minimum 0 and increment 1$"
		b := a[i:] // ERROR "Found redundant IsSliceInBounds$"
		x += b[0]
	}
	return x
}

func f0c(a []int) int {
	x := 0
	for i := range a { // ERROR "Induction variable with minimum 0 and increment 1$"
		b := a[:i+1] // ERROR "Found redundant IsSliceInBounds \(len promoted to cap\)$"
		x += b[0]
	}
	return x
}

func f1(a []int) int {
	x := 0
	for _, i := range a { // ERROR "Induction variable with minimum 0 and increment 1$"
		x += i
	}
	return x
}

func f2(a []int) int {
	x := 0
	for i := 1; i < len(a); i++ { // ERROR "Induction variable with minimum 1 and increment 1$"
		x += a[i] // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func f4(a [10]int) int {
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable with minimum 0 and increment 2$"
		x += a[i] // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func f5(a [10]int) int {
	x := 0
	for i := -10; i < len(a); i += 2 { // ERROR "Induction variable with minimum -10 and increment 2$"
		x += a[i]
	}
	return x
}

func f6(a []int) {
	for i := range a { // ERROR "Induction variable with minimum 0 and increment 1$"
		b := a[0:i] // ERROR "Found redundant IsSliceInBounds \(len promoted to cap\)$"
		f6(b)
	}
}

func g0a(a string) int {
	x := 0
	for i := 0; i < len(a); i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		x += int(a[i]) // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func g0b(a string) int {
	x := 0
	for i := 0; len(a) > i; i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		x += int(a[i]) // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func g1() int {
	a := "evenlength"
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable with minimum 0 and increment 2$"
		x += int(a[i]) // ERROR "Found redundant IsInBounds$"
	}
	return x
}

func g2() int {
	a := "evenlength"
	x := 0
	for i := 0; i < len(a); i += 2 { // ERROR "Induction variable with minimum 0 and increment 2$"
		j := i
		if a[i] == 'e' { // ERROR "Found redundant IsInBounds$"
			j = j + 1
		}
		x += int(a[j])
	}
	return x
}

func g3a() {
	a := "this string has length 25"
	for i := 0; i < len(a); i += 5 { // ERROR "Induction variable with minimum 0 and increment 5$"
		useString(a[i:]) // ERROR "Found redundant IsSliceInBounds$"
		useString(a[:i+3])
	}
}

func g3b(a string) {
	for i := 0; i < len(a); i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		useString(a[i+1:]) // ERROR "Found redundant IsSliceInBounds$"
	}
}

func g3c(a string) {
	for i := 0; i < len(a); i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		useString(a[:i+1]) // ERROR "Found redundant IsSliceInBounds$"
	}
}

func h1(a []byte) {
	c := a[:128]
	for i := range c { // ERROR "Induction variable with minimum 0 and increment 1$"
		c[i] = byte(i) // ERROR "Found redundant IsInBounds$"
	}
}

func h2(a []byte) {
	for i := range a[:128] { // ERROR "Induction variable with minimum 0 and increment 1$"
		a[i] = byte(i)
	}
}

func nobce1() {
	// tests overflow of max-min
	a := int64(9223372036854774057)
	b := int64(-1547)
	z := int64(1337)

	if a%z == b%z {
		panic("invalid test: modulos should differ")
	}

	for i := b; i < a; i += z {
		// No induction variable is possible because i will overflow a first iteration.
		useString("foobar")
	}
}

func nobce2(a string) {
	for i := int64(0); i < int64(len(a)); i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		useString(a[i:]) // ERROR "Found redundant IsSliceInBounds$"
	}
	for i := int64(0); i < int64(len(a))-31337; i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		useString(a[i:]) // ERROR "Found redundant IsSliceInBounds$"
	}
	for i := int64(0); i < int64(len(a))+int64(-1<<63); i++ { // ERROR "Induction variable with minimum 0 and increment 1$"
		// tests an overflow of StringLen-MinInt64
		useString(a[i:])
	}
}

//go:noinline
func useString(a string) {
}

func main() {
}
