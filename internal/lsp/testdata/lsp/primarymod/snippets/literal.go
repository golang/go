package snippets

import (
	"golang.org/x/tools/internal/lsp/signature"
	t "golang.org/x/tools/internal/lsp/types"
)

type structy struct {
	x signature.MyType
}

func X(_ map[signature.Alias]t.CoolAlias) (map[signature.Alias]t.CoolAlias) {
	return nil
}

func _() {
	X() //@signature(")", "X(_ map[signature.Alias]t.CoolAlias) map[signature.Alias]t.CoolAlias", 0)
	_ = signature.MyType{} //@item(literalMyType, "signature.MyType{}", "", "var")
	s := structy{
		x: //@snippet(" //", literalMyType, "signature.MyType{\\}", "signature.MyType{\\}")
	}
}