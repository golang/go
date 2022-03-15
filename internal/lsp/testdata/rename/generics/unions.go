//go:build go1.18
// +build go1.18

package generics

type T string //@rename("T", "R")

type C interface {
	T | ~int //@rename("T", "S")
}
