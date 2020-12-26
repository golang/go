// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package p

type T int

func (T) Mv()  {}
func (*T) Mp() {}

type P1 struct{ T }
type P2 struct{ *T }
type P3 *struct{ T }
type P4 *struct{ *T }

func _() {
	{
		var p P1
		p.Mv()
		(&p).Mv()
		(*&p).Mv()
		p.Mp()
		(&p).Mp()
		(*&p).Mp()
	}
	{
		var p P2
		p.Mv()
		(&p).Mv()
		(*&p).Mv()
		p.Mp()
		(&p).Mp()
		(*&p).Mp()
	}
	{
		var p P3
		p.Mv()     // ERROR "undefined"
		(&p).Mv()  // ERROR "undefined"
		(*&p).Mv() // ERROR "undefined"
		(**&p).Mv()
		(*p).Mv()
		(&*p).Mv()
		p.Mp()     // ERROR "undefined"
		(&p).Mp()  // ERROR "undefined"
		(*&p).Mp() // ERROR "undefined"
		(**&p).Mp()
		(*p).Mp()
		(&*p).Mp()
	}
	{
		var p P4
		p.Mv()     // ERROR "undefined"
		(&p).Mv()  // ERROR "undefined"
		(*&p).Mv() // ERROR "undefined"
		(**&p).Mv()
		(*p).Mv()
		(&*p).Mv()
		p.Mp()     // ERROR "undefined"
		(&p).Mp()  // ERROR "undefined"
		(*&p).Mp() // ERROR "undefined"
		(**&p).Mp()
		(*p).Mp()
		(&*p).Mp()
	}
}

func _() {
	type P5 struct{ T }
	type P6 struct{ *T }
	type P7 *struct{ T }
	type P8 *struct{ *T }

	{
		var p P5
		p.Mv()
		(&p).Mv()
		(*&p).Mv()
		p.Mp()
		(&p).Mp()
		(*&p).Mp()
	}
	{
		var p P6
		p.Mv()
		(&p).Mv()
		(*&p).Mv()
		p.Mp()
		(&p).Mp()
		(*&p).Mp()
	}
	{
		var p P7
		p.Mv()     // ERROR "undefined"
		(&p).Mv()  // ERROR "undefined"
		(*&p).Mv() // ERROR "undefined"
		(**&p).Mv()
		(*p).Mv()
		(&*p).Mv()
		p.Mp()     // ERROR "undefined"
		(&p).Mp()  // ERROR "undefined"
		(*&p).Mp() // ERROR "undefined"
		(**&p).Mp()
		(*p).Mp()
		(&*p).Mp()
	}
	{
		var p P8
		p.Mv()     // ERROR "undefined"
		(&p).Mv()  // ERROR "undefined"
		(*&p).Mv() // ERROR "undefined"
		(**&p).Mv()
		(*p).Mv()
		(&*p).Mv()
		p.Mp()     // ERROR "undefined"
		(&p).Mp()  // ERROR "undefined"
		(*&p).Mp() // ERROR "undefined"
		(**&p).Mp()
		(*p).Mp()
		(&*p).Mp()
	}
}
