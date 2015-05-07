// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./other"

type Imported interface {
	Do()
}

type HasAMethod struct {
	x int
}

func (me *HasAMethod) Do() {
	println(me.x)
}

func InMyCode(x *Imported, y *HasAMethod, z *other.Exported) {
	x.Do() // ERROR "x\.Do undefined \(type \*Imported is pointer to interface, not interface\)"
	x.do() // ERROR "x\.do undefined \(type \*Imported is pointer to interface, not interface\)"
	(*x).Do()
	x.Dont()    // ERROR "x\.Dont undefined \(type \*Imported is pointer to interface, not interface\)"
	(*x).Dont() // ERROR "\(\*x\)\.Dont undefined \(type Imported has no field or method Dont\)"

	y.Do()
	y.do() // ERROR "y\.do undefined \(type \*HasAMethod has no field or method do, but does have Do\)"
	(*y).Do()
	(*y).do()   // ERROR "\(\*y\)\.do undefined \(type HasAMethod has no field or method do, but does have Do\)"
	y.Dont()    // ERROR "y\.Dont undefined \(type \*HasAMethod has no field or method Dont\)"
	(*y).Dont() // ERROR "\(\*y\)\.Dont undefined \(type HasAMethod has no field or method Dont\)"

	z.Do() // ERROR "z\.Do undefined \(type \*other\.Exported is pointer to interface, not interface\)"
	z.do() // ERROR "z\.do undefined \(type \*other\.Exported is pointer to interface, not interface\)"
	(*z).Do()
	(*z).do()     // ERROR "\(\*z\)\.do undefined \(type other.Exported has no field or method do, but does have Do\)"
	z.Dont()      // ERROR "z\.Dont undefined \(type \*other\.Exported is pointer to interface, not interface\)"
	(*z).Dont()   // ERROR "\(\*z\)\.Dont undefined \(type other\.Exported has no field or method Dont\)"
	z.secret()    // ERROR "z\.secret undefined \(type \*other\.Exported is pointer to interface, not interface\)"
	(*z).secret() // ERROR "\(\*z\)\.secret undefined \(cannot refer to unexported field or method secret\)"

}

func main() {
}
