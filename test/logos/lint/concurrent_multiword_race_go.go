// errorcheck

package p

import "logos/sync"

// Go twin of concurrent_multiword_race.lgo: the go statement leg of
// concurrency::concurrent_multiword_race (SPEC Sec 1.2, #13, #148) on a Go
// package, where it runs at deny. The group child, the union, and the const
// view of the .lgo file have no Go form; the logos/sync exemption stays.

type Pair struct {
	A, B int
}

type Triple struct {
	A, B, C int
}

type OneWord struct {
	N int
}

type Padded struct {
	N int
	Z struct{}
}

type Named struct {
	Name string
	N    int
}

type Holder struct {
	P *Named
	N int
}

type TwoBytes struct {
	A, B byte
}

func stringWrittenByParent() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	s = "b"
}

func stringWrittenByChild() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		s = "b"
	}()
	println(len(s))
}

func sliceHeaderWritten() {
	xs := []int{1}
	go func() { // ERROR "concurrent_multiword_race"
		xs = append(xs, 2)
	}()
}

func interfaceWritten() {
	var v any = 1
	go func() { // ERROR "concurrent_multiword_race"
		println(v != nil)
	}()
	v = "x"
}

func complexWritten() {
	c := complex(1, 2)
	go func() { // ERROR "concurrent_multiword_race"
		c = complex(3, 4)
	}()
	println(real(c))
}

func structFieldWritten() {
	var n Named
	go func() { // ERROR "concurrent_multiword_race"
		println(n.N)
	}()
	n.Name = "x"
}

func arrayElementWritten() {
	var a [2]int
	go func() { // ERROR "concurrent_multiword_race"
		a[0] = 1
	}()
}

func compoundAssignWritten() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	s += "b"
}

func rangeAssignWritten() {
	var s string
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	for _, s = range []string{"a"} {
	}
}

func redeclaredWritten() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	s, n := "b", 1
	println(n)
}

func writtenAfterOtherStatements() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	println("work")
	if len(s) > 0 {
		s = "b"
	}
}

func parenthesizedLiteral() {
	s := "a"
	go (func() { // ERROR "concurrent_multiword_race"
		s = "b"
	})()
	println(s)
}

func siblingWrites() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	go func() { // ERROR "concurrent_multiword_race"
		s = "b"
	}()
}

func nestedLiteralWrites() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		println(s)
	}()
	func() {
		s = "b"
	}()
}

func spawnInsideChild() {
	s := "a"
	go func() { // ERROR "concurrent_multiword_race"
		go func() { // ERROR "concurrent_multiword_race"
			s = "b"
		}()
	}()
	println(s)
}

func boundaryTwoWordStruct() {
	var p Pair
	go func() { // ERROR "concurrent_multiword_race"
		p.A = 1
	}()
}

func boundaryThreeWordStruct() {
	var t Triple
	go func() { // ERROR "concurrent_multiword_race"
		t.C = 1
	}()
}

func boundaryTwoWordArray() {
	var a [2]int
	go func() { // ERROR "concurrent_multiword_race"
		a = [2]int{1, 2}
	}()
	println(a[0])
}

func boundaryThreeWordArray() {
	var a [3]int
	go func() { // ERROR "concurrent_multiword_race"
		a = [3]int{1, 2, 3}
	}()
	println(a[0])
}

func boundaryOneElementStringArray() {
	var a [1]string
	go func() { // ERROR "concurrent_multiword_race"
		a[0] = "x"
	}()
}

func boundaryOneWordStruct() {
	var o OneWord
	go func() {
		o.N = 1
	}()
}

func boundaryPaddedStruct() {
	var p Padded
	go func() {
		p.N = 1
	}()
}

func boundaryOneWordArray() {
	var a [1]int
	go func() {
		a[0] = 1
	}()
}

func boundaryEmptyArray() {
	var a [0]string
	go func() {
		a = [0]string{}
	}()
	println(len(a))
}

func passTwoByteStruct() {
	var s TwoBytes
	go func() {
		s.A = 1
	}()
	s.B = 2
	println(s.A)
}

func passTwoByteArray() {
	var a [2]byte
	go func() {
		a[0] = 1
	}()
	a[1] = 2
	println(a[0])
}

func passIntWritten() {
	n := 0
	go func() {
		n = 1
	}()
	n = 2
	println(n)
}

func passPointerWritten() {
	x := 0
	p := &x
	go func() {
		p = nil
	}()
	p = &x
	println(p != nil)
}

func passMapWritten() {
	m := map[string]int{}
	go func() {
		m = nil
	}()
	m = map[string]int{}
	println(len(m))
}

func passReadOnlyBothSides() {
	s := "a"
	go func() {
		println(s)
	}()
	println(s)
}

func passWrittenBeforeSpawn() {
	s := "a"
	s = "b"
	go func() {
		println(s)
	}()
}

func passSliceElementWritten() {
	xs := []int{1, 2}
	go func() {
		xs[0] = 1
	}()
	xs[1] = 2
}

func passDeclaredInChild() {
	go func() {
		s := "a"
		s = "b"
		println(s)
	}()
}

func passShadowedInChild() {
	s := "a"
	go func() {
		s := "x"
		s = "y"
		println(s)
	}()
	s = "b"
	println(s)
}

func passParameterOfLiteral() {
	s := "a"
	go func(s string) {
		s = "b"
	}(s)
	s = "c"
}

func use(s string) {
	println(s)
}

func passArgumentCopied() {
	s := "a"
	go use(s)
	s = "b"
}

func passPackageVariable() {
	go func() {
		global = "b"
	}()
	global = "c"
}

var global string

func passLockedValue() {
	var l sync.Locked[string]
	go func() {
		l = sync.Locked[string]{}
	}()
	l = sync.Locked[string]{}
	l.WithLock(func(s *string) { println(*s) })
}

func passLockedUsedThroughWithLock() {
	var l sync.Locked[string]
	go func() {
		l.WithLock(func(s *string) { *s = "a" })
	}()
	l.WithLock(func(s *string) { *s = "b" })
}

func passHotValue() {
	var h sync.Hot[string]
	go func() {
		h = sync.Hot[string]{}
	}()
	h = sync.Hot[string]{}
	println(h.Load() == nil)
}

func passNoSpawn() {
	s := "a"
	func() {
		s = "b"
	}()
	println(s)
}

func passWriteThroughPointerField() {
	h := Holder{P: &Named{}}
	go func() {
		h.P.Name = "a"
	}()
	h.P.Name = "b"
}

func passFieldOfPointer(p *Named) {
	go func() {
		p.Name = "a"
	}()
	p.Name = "b"
}
