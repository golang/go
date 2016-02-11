package main

// This is a test of 'pointsto', but we split it into a separate file
// so that pointsto.go doesn't have to import "reflect" each time.

import "reflect"

var a int
var b bool

func main() {
	m := make(map[*int]*bool)
	m[&a] = &b

	mrv := reflect.ValueOf(m)
	if a > 0 {
		mrv = reflect.ValueOf(&b)
	}
	if a > 0 {
		mrv = reflect.ValueOf(&a)
	}

	_ = mrv                  // @pointsto mrv "mrv"
	p1 := mrv.Interface()    // @pointsto p1 "p1"
	p2 := mrv.MapKeys()      // @pointsto p2 "p2"
	p3 := p2[0]              // @pointsto p3 "p3"
	p4 := reflect.TypeOf(p1) // @pointsto p4 "p4"

	_, _, _, _ = p1, p2, p3, p4
}
