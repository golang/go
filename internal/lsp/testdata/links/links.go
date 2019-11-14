package links

import (
	"fmt" //@link(`fmt`,"https://godoc.org/fmt")

	"golang.org/x/tools/internal/lsp/foo" //@link(`golang.org/x/tools/internal/lsp/foo`,`https://godoc.org/golang.org/x/tools/internal/lsp/foo`)

	_ "database/sql" //@link(`database/sql`, `https://godoc.org/database/sql`)

	errors "golang.org/x/xerrors" //@link(`golang.org/x/xerrors`, `https://godoc.org/golang.org/x/xerrors`)
)

var (
	_ fmt.Formatter
	_ foo.StructFoo
	_ errors.Formatter
)

// Foo function
func Foo() string {
	/*https://example.com/comment */ //@link("https://example.com/comment","https://example.com/comment")

	url := "https://example.com/string_literal" //@link("https://example.com/string_literal","https://example.com/string_literal")
	return url
}
