package links

import (
	"fmt" //@link(`fmt`,"https://pkg.go.dev/fmt")

	"golang.org/x/tools/internal/lsp/foo" //@link(`golang.org/x/tools/internal/lsp/foo`,`https://pkg.go.dev/golang.org/x/tools/internal/lsp/foo`)

	_ "database/sql" //@link(`database/sql`, `https://pkg.go.dev/database/sql`)

	_ "example.com/extramodule/pkg" //@link(`example.com/extramodule/pkg`,`https://pkg.go.dev/example.com/extramodule@v1.0.0/pkg`)
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

	// TODO(golang/go#1234): Link the relevant issue. //@link("golang/go#1234", "https://github.com/golang/go/issues/1234")
	// TODO(microsoft/vscode-go#12): Another issue. //@link("microsoft/vscode-go#12", "https://github.com/microsoft/vscode-go/issues/12")
}
