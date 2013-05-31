package main

// Test of promotion of methods of an interface embedded within a
// struct.  In particular, this test excercises that the correct
// method is called.

type I interface {
	one() int
	two() string
}

type S struct {
	I
}

type impl struct{}

func (impl) one() int {
	return 1
}

func (impl) two() string {
	return "two"
}

func main() {
	var s S
	s.I = impl{}
	if one := s.one(); one != 1 {
		panic(one)
	}
	if two := s.two(); two != "two" {
		panic(two)
	}
}
