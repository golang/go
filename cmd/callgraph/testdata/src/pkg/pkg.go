package main

type I interface {
	f()
}

type C int

func (C) f() {}

type D int

func (D) f() {}

func main() {
	var i I = C(0)
	i.f() // dynamic call

	main2()
}

func main2() {
	var i I = D(0)
	i.f() // dynamic call
}
