package nodisk

import (
	"golang.org/x/tools/internal/lsp/foo"
)

func _() {
	foo.Foo() //@complete("F", IntFoo, StructFoo, Foo)
}
