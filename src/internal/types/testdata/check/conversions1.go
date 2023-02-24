// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test various valid and invalid struct assignments and conversions.
// Does not compile.

package conversions2

type I interface {
	m()
}

// conversions between structs

func _() {
	type S struct{}
	type T struct{}
	var s S
	var t T
	var u struct{}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u
	s = S(s)
	s = S(t)
	s = S(u)
	t = u
	t = T(u)
}

func _() {
	type S struct{ x int }
	type T struct {
		x int "foo"
	}
	var s S
	var t T
	var u struct {
		x int "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = S(s)
	s = S(t)
	s = S(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = T(u)
}

func _() {
	type E struct{ x int }
	type S struct{ x E }
	type T struct {
		x E "foo"
	}
	var s S
	var t T
	var u struct {
		x E "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = S(s)
	s = S(t)
	s = S(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = T(u)
}

func _() {
	type S struct {
		x struct {
			x int "foo"
		}
	}
	type T struct {
		x struct {
			x int "bar"
		} "foo"
	}
	var s S
	var t T
	var u struct {
		x struct {
			x int "bar"
		} "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = S(s)
	s = S(t)
	s = S(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = T(u)
}

func _() {
	type E1 struct {
		x int "foo"
	}
	type E2 struct {
		x int "bar"
	}
	type S struct{ x E1 }
	type T struct {
		x E2 "foo"
	}
	var s S
	var t T
	var u struct {
		x E2 "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = S(s)
	s = S(t /* ERROR "cannot convert" */ )
	s = S(u /* ERROR "cannot convert" */ )
	t = u   // ERRORx `cannot use .* in assignment`
	t = T(u)
}

func _() {
	type E struct{ x int }
	type S struct {
		f func(struct {
			x int "foo"
		})
	}
	type T struct {
		f func(struct {
			x int "bar"
		})
	}
	var s S
	var t T
	var u struct{ f func(E) }
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = S(s)
	s = S(t)
	s = S(u /* ERROR "cannot convert" */ )
	t = u   // ERRORx `cannot use .* in assignment`
	t = T(u /* ERROR "cannot convert" */ )
}

// conversions between pointers to structs

func _() {
	type S struct{}
	type T struct{}
	var s *S
	var t *T
	var u *struct{}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = (*T)(u)
}

func _() {
	type S struct{ x int }
	type T struct {
		x int "foo"
	}
	var s *S
	var t *T
	var u *struct {
		x int "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = (*T)(u)
}

func _() {
	type E struct{ x int }
	type S struct{ x E }
	type T struct {
		x E "foo"
	}
	var s *S
	var t *T
	var u *struct {
		x E "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = (*T)(u)
}

func _() {
	type S struct {
		x struct {
			x int "foo"
		}
	}
	type T struct {
		x struct {
			x int "bar"
		} "foo"
	}
	var s *S
	var t *T
	var u *struct {
		x struct {
			x int "bar"
		} "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u)
	t = u // ERRORx `cannot use .* in assignment`
	t = (*T)(u)
}

func _() {
	type E1 struct {
		x int "foo"
	}
	type E2 struct {
		x int "bar"
	}
	type S struct{ x E1 }
	type T struct {
		x E2 "foo"
	}
	var s *S
	var t *T
	var u *struct {
		x E2 "bar"
	}
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t /* ERROR "cannot convert" */ )
	s = (*S)(u /* ERROR "cannot convert" */ )
	t = u      // ERRORx `cannot use .* in assignment`
	t = (*T)(u)
}

func _() {
	type E struct{ x int }
	type S struct {
		f func(struct {
			x int "foo"
		})
	}
	type T struct {
		f func(struct {
			x int "bar"
		})
	}
	var s *S
	var t *T
	var u *struct{ f func(E) }
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u /* ERROR "cannot convert" */ )
	t = u      // ERRORx `cannot use .* in assignment`
	t = (*T)(u /* ERROR "cannot convert" */ )
}

func _() {
	type E struct{ x int }
	type S struct {
		f func(*struct {
			x int "foo"
		})
	}
	type T struct {
		f func(*struct {
			x int "bar"
		})
	}
	var s *S
	var t *T
	var u *struct{ f func(E) }
	s = s
	s = t // ERRORx `cannot use .* in assignment`
	s = u // ERRORx `cannot use .* in assignment`
	s = (*S)(s)
	s = (*S)(t)
	s = (*S)(u /* ERROR "cannot convert" */ )
	t = u      // ERRORx `cannot use .* in assignment`
	t = (*T)(u /* ERROR "cannot convert" */ )
}
