package testdata

import "sync"

func OkFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
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
}
