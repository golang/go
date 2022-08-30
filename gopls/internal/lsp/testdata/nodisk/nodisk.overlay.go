package nodisk

import (
	"golang.org/lsptests/foo"
)

func _() {
	foo.Foo() //@complete("F", Foo, IntFoo, StructFoo)
}
