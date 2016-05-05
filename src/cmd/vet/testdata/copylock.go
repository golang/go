package testdata

import "sync"

func OkFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y

	var z = sync.Mutex{}
	w := sync.Mutex{}

	w = sync.Mutex{}
	q := struct{ L sync.Mutex }{
		L: sync.Mutex{},
	}

	yy := []Tlock{
		Tlock{},
		Tlock{
			once: sync.Once{},
		},
	}

	nl := new(sync.Mutex)
	mx := make([]sync.Mutex, 10)
	xx := struct{ L *sync.Mutex }{
		L: new(sync.Mutex),
	}
}

type Tlock struct {
	once sync.Once
}

func BadFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
	*p = *x // ERROR "assignment copies lock value to \*p: sync.Mutex"

	var t Tlock
	var tp *Tlock
	tp = &t
	*tp = t // ERROR "assignment copies lock value to \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	t = *tp // ERROR "assignment copies lock value to t: testdata.Tlock contains sync.Once contains sync.Mutex"

	y := *x   // ERROR "assignment copies lock value to y: sync.Mutex"
	var z = t // ERROR "variable declaration copies lock value to z: testdata.Tlock contains sync.Once contains sync.Mutex"

	w := struct{ L sync.Mutex }{
		L: *x, // ERROR "literal copies lock value from \*x: sync.Mutex"
	}
	var q = map[int]Tlock{
		1: t,   // ERROR "literal copies lock value from t: testdata.Tlock contains sync.Once contains sync.Mutex"
		2: *tp, // ERROR "literal copies lock value from \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	}
	yy := []Tlock{
		t,   // ERROR "literal copies lock value from t: testdata.Tlock contains sync.Once contains sync.Mutex"
		*tp, // ERROR "literal copies lock value from \*tp: testdata.Tlock contains sync.Once contains sync.Mutex"
	}

	// override 'new' keyword
	new := func(interface{}) {}
	new(t) // ERROR "function call copies lock value: testdata.Tlock contains sync.Once contains sync.Mutex"
}
