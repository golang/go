//go:build go1.18
// +build go1.18

package generics

type foo[P any] int //@rename("foo","bar")

var x struct{ foo[int] }

var _ = x.foo
