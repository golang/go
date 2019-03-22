package b_x_test

import (
	"a"
	"b"
)

func ExampleFoo_F() {
	var x b.Foo
	x.F()
	a.Foo()
}

func ExampleFoo_G() { // want "ExampleFoo_G refers to unknown field or method: Foo.G"

}

func ExampleBar_F() { // want "ExampleBar_F refers to unknown identifier: Bar"

}
