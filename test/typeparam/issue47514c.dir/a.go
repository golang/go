package a

type Doer[T any] interface {
	Do() T
}
