// want package:`features{typeDecl,funcDecl,funcInstance}`

package a

type T[P any] int

func F[P any]() {}

var _ = F[int]
