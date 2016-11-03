package alias // @describe pkg "alias"

// +build go1.8

// Test describe queries on Go 1.8 aliases.
// See go.tools/guru/guru_test.go for explanation.
// See alias.golden for expected query results.

import (
	"aliaslib"
	"nosuchpkg"
)

var bad1 => nopkg.NoVar// @describe bad1 "bad1"
var bad2 => nosuchpkg.NoVar// @describe bad2 "bad2"

var v_ => aliaslib.V // @describe v "v_"
type t_ => aliaslib.T // @describe t "t_"
const c_ => aliaslib.C // @describe c "c_"
func f_ => aliaslib.F // @describe f "f_"

type S1 struct { aliaslib.T } // @describe s1-field "T"
type S2 struct { t_ } // @describe s2-field "t_"

var x t_ // @describe var-x "t_"
