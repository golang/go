package highlights

import "fmt"

type F struct{ bar int }

var foo = F{bar: 52} //@highlight("foo", "foo")

func Print() {
	fmt.Println(foo) //@highlight("foo", "foo")
}

func (x *F) Inc() { //@highlight("x", "x")
	x.bar++ //@highlight("x", "x")
}
