package signature_test

import (
	"testing"

	sig "golang.org/x/tools/internal/lsp/signature"
)

func TestSignature(t *testing.T) {
	sig.AliasSlice()    //@signature(")", "AliasSlice(a []*sig.Alias) (b sig.Alias)", 0)
	sig.AliasMap()      //@signature(")", "AliasMap(a map[*sig.Alias]sig.StringAlias) (b map[*sig.Alias]sig.StringAlias, c map[*sig.Alias]sig.StringAlias)", 0)
	sig.OtherAliasMap() //@signature(")", "OtherAliasMap(a map[sig.Alias]sig.OtherAlias, b map[sig.Alias]sig.OtherAlias) map[sig.Alias]sig.OtherAlias", 0)
}
