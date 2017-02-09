// Tests of Go 1.9 type aliases.
// See go.tools/guru/guru_test.go for explanation.
// See alias.golden for expected query results.

package alias // @describe describe-pkg "alias"

type I interface{ f() } // @implements implements-I "I"

type N int
func (N) f() {}

type M = N // @describe describe-def-M "M"
var m M // @describe describe-ref-M "M"

type O N // @describe describe-O "O"

type P = struct{N} // @describe describe-P "N"

type U = undefined // @describe describe-U "U"
type _ = undefined // @describe describe-undefined "undefined"
