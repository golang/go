package copylock

import "sync"

func BadFunc() {
	var x *sync.Mutex
	p := x
	var y sync.Mutex
	p = &y
	*p = *x // ERROR "assignment copies lock value to \*p: sync.Mutex"
}
