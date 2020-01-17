package source

import (
	"go/parser"
	"go/token"
	"testing"
)

func TestTrimToImports(t *testing.T) {
	const input = `package source

import (
	m
	"fmt"
)

func foo() {
	fmt.Println("hi")
}
`

	fs := token.NewFileSet()
	f, _ := parser.ParseFile(fs, "foo.go", input, parser.ImportsOnly)
	trimToImports(fs, f, []byte(input))
}
