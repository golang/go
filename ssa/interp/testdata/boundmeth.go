// Tests of bound method closures.

package main

func assert(b bool) {
	if !b {
		panic("oops")
	}
}

type I int

func (i I) add(x int) int {
	return int(i) + x
}

func valueReceiver() {
	var three I = 3
	assert(three.add(5) == 8)
	var add3 func(int) int = three.add
	assert(add3(5) == 8)
}

type S struct{ x int }

func (s *S) incr() {
	s.x++
}

func (s *S) get() int {
	return s.x
}

func pointerReceiver() {
	ps := new(S)
	incr := ps.incr
	get := ps.get
	assert(get() == 0)
	incr()
	incr()
	incr()
	assert(get() == 3)
}

func addressibleValuePointerReceiver() {
	var s S
	incr := s.incr
	get := s.get
	assert(get() == 0)
	incr()
	incr()
	incr()
	assert(get() == 3)
}

type S2 struct {
	S
}

func promotedReceiver() {
	var s2 S2
	incr := s2.incr
	get := s2.get
	assert(get() == 0)
	incr()
	incr()
	incr()
	assert(get() == 3)
}

func anonStruct() {
	var s struct{ S }
	incr := s.incr
	get := s.get
	assert(get() == 0)
	incr()
	incr()
	incr()
	assert(get() == 3)
}

func typeCheck() {
	var i interface{}
	i = (*S).incr
	_ = i.(func(*S)) // type assertion: receiver type prepended to params

	var s S
	i = s.incr
	_ = i.(func()) // type assertion: receiver type disappears
}

type errString string

func (err errString) Error() string {
	return string(err)
}

// Regression test for a builder crash.
func regress1(x error) func() string {
	return x.Error
}

func main() {
	valueReceiver()
	pointerReceiver()
	addressibleValuePointerReceiver()
	promotedReceiver()
	anonStruct()
	typeCheck()

	if e := regress1(errString("hi"))(); e != "hi" {
		panic(e)
	}
}
