package snippets

import (
	"golang.org/x/tools/internal/lsp/signature"
	"golang.org/x/tools/internal/lsp/types"
)

type structy struct {
	x signature.MyType
}

func X(_ map[signatures.Alias]types.CoolAlias) (map[signatures.Alias]types.CoolAlias) {
	return nil
}

func _() {
	X() //@signature(")", "X(_ map[signatures.Alias]types.CoolAlias) map[signatures.Alias]types.CoolAlias", 0)
	_ = signature.MyType{} //@item(literalMyType, "signature.MyType{}", "", "var")
	s := structy{
		x: //@snippet(" //", literalMyType, "signature.MyType{\\}", "signature.MyType{\\}")
	}
}