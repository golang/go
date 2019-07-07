package links

import (
	"fmt" //@link(re`".*"`,"https://godoc.org/fmt")

	"golang.org/x/tools/internal/lsp/foo" //@link(re`".*"`,`https://godoc.org/golang.org/x/tools/internal/lsp/foo`)
)

var (
	_ fmt.Formatter
	_ foo.StructFoo
)

// Foo function
func Foo() string {
	/*https://example.com/comment */ //@link("https://example.com/comment","https://example.com/comment")
	url := "https://example.com/string_literal" //@link("https://example.com/string_literal","https://example.com/string_literal")
	return url
}
