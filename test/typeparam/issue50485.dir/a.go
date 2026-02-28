package a

import "fmt"

type ImplicitOrd interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}

func LessGiven[T ImplicitOrd]() Ord[T] {
	return LessFunc[T](func(a, b T) bool {
		return a < b
	})
}

type Eq[T any] interface {
	Eqv(a T, b T) bool
}

type Ord[T any] interface {
	Eq[T]
	Less(a T, b T) bool
}

type LessFunc[T any] func(a, b T) bool

func (r LessFunc[T]) Eqv(a, b T) bool {
	return r(a, b) == false && r(b, a) == false
}

func (r LessFunc[T]) Less(a, b T) bool {
	return r(a, b)
}

type Option[T any] struct {
	v *T
}

func (r Option[T]) IsDefined() bool {
	return r.v != nil
}

func (r Option[T]) IsEmpty() bool {
	return !r.IsDefined()
}

func (r Option[T]) Get() T {
	return *r.v
}

func (r Option[T]) String() string {
	if r.IsDefined() {
		return fmt.Sprintf("Some(%v)", r.v)
	} else {
		return "None"
	}
}

func (r Option[T]) OrElse(t T) T {
	if r.IsDefined() {
		return *r.v
	}
	return t
}

func (r Option[T]) Recover(f func() T) Option[T] {
	if r.IsDefined() {
		return r
	}
	t := f()
	return Option[T]{&t}
}

type Func1[A1, R any] func(a1 A1) R

type Func2[A1, A2, R any] func(a1 A1, a2 A2) R

func (r Func2[A1, A2, R]) Curried() Func1[A1, Func1[A2, R]] {
	return func(a1 A1) Func1[A2, R] {
		return Func1[A2, R](func(a2 A2) R {
			return r(a1, a2)
		})
	}
}

type HList interface {
	sealed()
}

// Header is constrains interface type,  enforce Head type of Cons is HT
type Header[HT any] interface {
	HList
	Head() HT
}

// Cons means H :: T
// zero value of Cons[H,T] is not allowed.
// so Cons defined as interface type
type Cons[H any, T HList] interface {
	HList
	Head() H
	Tail() T
}

type Nil struct {
}

func (r Nil) Head() Nil {
	return r
}

func (r Nil) Tail() Nil {
	return r
}

func (r Nil) String() string {
	return "Nil"
}

func (r Nil) sealed() {

}

type hlistImpl[H any, T HList] struct {
	head H
	tail T
}

func (r hlistImpl[H, T]) Head() H {
	return r.head
}

func (r hlistImpl[H, T]) Tail() T {
	return r.tail
}

func (r hlistImpl[H, T]) String() string {
	return fmt.Sprintf("%v :: %v", r.head, r.tail)
}

func (r hlistImpl[H, T]) sealed() {

}

func hlist[H any, T HList](h H, t T) Cons[H, T] {
	return hlistImpl[H, T]{h, t}
}

func Concat[H any, T HList](h H, t T) Cons[H, T] {
	return hlist(h, t)
}

func Empty() Nil {
	return Nil{}
}
func Some[T any](v T) Option[T] {
	return Option[T]{}.Recover(func() T {
		return v
	})
}

func None[T any]() Option[T] {
	return Option[T]{}
}

func Ap[T, U any](t Option[Func1[T, U]], a Option[T]) Option[U] {
	return FlatMap(t, func(f Func1[T, U]) Option[U] {
		return Map(a, f)
	})
}

func Map[T, U any](opt Option[T], f func(v T) U) Option[U] {
	return FlatMap(opt, func(v T) Option[U] {
		return Some(f(v))
	})
}

func FlatMap[T, U any](opt Option[T], fn func(v T) Option[U]) Option[U] {
	if opt.IsDefined() {
		return fn(opt.Get())
	}
	return None[U]()
}

type ApplicativeFunctor1[H Header[HT], HT, A, R any] struct {
	h  Option[H]
	fn Option[Func1[A, R]]
}

func (r ApplicativeFunctor1[H, HT, A, R]) ApOption(a Option[A]) Option[R] {
	return Ap(r.fn, a)
}

func (r ApplicativeFunctor1[H, HT, A, R]) Ap(a A) Option[R] {
	return r.ApOption(Some(a))
}

func Applicative1[A, R any](fn Func1[A, R]) ApplicativeFunctor1[Nil, Nil, A, R] {
	return ApplicativeFunctor1[Nil, Nil, A, R]{Some(Empty()), Some(fn)}
}

type ApplicativeFunctor2[H Header[HT], HT, A1, A2, R any] struct {
	h  Option[H]
	fn Option[Func1[A1, Func1[A2, R]]]
}

func (r ApplicativeFunctor2[H, HT, A1, A2, R]) ApOption(a Option[A1]) ApplicativeFunctor1[Cons[A1, H], A1, A2, R] {

	nh := FlatMap(r.h, func(hv H) Option[Cons[A1, H]] {
		return Map(a, func(av A1) Cons[A1, H] {
			return Concat(av, hv)
		})
	})

	return ApplicativeFunctor1[Cons[A1, H], A1, A2, R]{nh, Ap(r.fn, a)}
}
func (r ApplicativeFunctor2[H, HT, A1, A2, R]) Ap(a A1) ApplicativeFunctor1[Cons[A1, H], A1, A2, R] {

	return r.ApOption(Some(a))
}

func Applicative2[A1, A2, R any](fn Func2[A1, A2, R]) ApplicativeFunctor2[Nil, Nil, A1, A2, R] {
	return ApplicativeFunctor2[Nil, Nil, A1, A2, R]{Some(Empty()), Some(fn.Curried())}
}
func OrdOption[T any](m Ord[T]) Ord[Option[T]] {
	return LessFunc[Option[T]](func(t1 Option[T], t2 Option[T]) bool {
		if !t1.IsDefined() && !t2.IsDefined() {
			return false
		}
		return Applicative2(m.Less).ApOption(t1).ApOption(t2).OrElse(!t1.IsDefined())
	})
}

func Given[T ImplicitOrd]() Ord[T] {
	return LessGiven[T]()
}
