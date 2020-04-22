package signature_test

import (
	"testing"

	"golang.org/x/tools/internal/lsp/signature"
)

func TestSignature(t *testing.T) {
	signature.AliasSlice()    //@signature(")", "AliasSlice(a []*signature.Alias) (b signature.Alias)", 0)
	signature.AliasMap()      //@signature(")", "AliasMap(a map[*signature.Alias]signature.StringAlias) (b map[*signature.Alias]signature.StringAlias, c map[*signature.Alias]signature.StringAlias)", 0)
	signature.OtherAliasMap() //@signature(")", "OtherAliasMap(a map[signature.Alias]signature.OtherAlias, b map[signature.Alias]signature.OtherAlias) map[signature.Alias]signature.OtherAlias", 0)
}
