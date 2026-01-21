// compile

package p

type S struct{ f func(int) }

func g[T any](T) {}

func _() {
	var s S
	s.f = g      // ok
	_ = S{f: g}  // should be ok; was rejected
	_ = s
}
