// +build ignore

package main

var unknown bool

type S string

func incr(x int) int { return x + 1 }

func main() {
	var i interface{}
	i = 1
	if unknown {
		i = S("foo")
	}
	if unknown {
		i = (func(int, int))(nil) // NB type compares equal to that below.
	}
	// Look, the test harness can handle equal-but-not-String-equal
	// types because we parse types and using a typemap.
	if unknown {
		i = (func(x int, y int))(nil)
	}
	if unknown {
		i = incr
	}
	print(i) // @types int | S | func(int, int) | func(int) int

	// NB, an interface may never directly alias any global
	// labels, even though it may contain pointers that do.
	print(i)                 // @pointsto makeinterface:func(x int) int | makeinterface:func(x int, y int) | makeinterface:func(int, int) | makeinterface:int | makeinterface:main.S
	print(i.(func(int) int)) // @pointsto main.incr
}
