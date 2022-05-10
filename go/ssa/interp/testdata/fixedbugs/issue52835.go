package main

var called bool

type I interface {
	Foo()
}

type A struct{}

func (a A) Foo() {
	called = true
}

func lambda[X I]() func() func() {
	return func() func() {
		var x X
		return x.Foo
	}
}

func main() {
	lambda[A]()()()
	if !called {
		panic(called)
	}
}
