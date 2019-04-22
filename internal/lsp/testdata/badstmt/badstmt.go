package badstmt

import (
	"golang.org/x/tools/internal/lsp/foo"
)

func _() {
	defer foo.F //@complete("F", Foo, IntFoo, StructFoo),diag(" //", "LSP", "function must be invoked in defer statement")
	go foo.F //@complete("F", Foo, IntFoo, StructFoo)
}