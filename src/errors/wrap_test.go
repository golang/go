// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"reflect"
	"testing"
)

func TestIs(t *testing.T) {
	err1 := errors.New("1")
	erra := wrapped{"wrap 2", err1}
	errb := wrapped{"wrap 3", erra}

	err3 := errors.New("3")

	poser := &poser{"either 1 or 3", func(err error) bool {
		return err == err1 || err == err3
	}}

	testCases := []struct {
		err    error
		target error
		match  bool
	}{
		{nil, nil, true},
		{nil, err1, false},
		{err1, nil, false},
		{err1, err1, true},
		{erra, err1, true},
		{errb, err1, true},
		{err1, err3, false},
		{erra, err3, false},
		{errb, err3, false},
		{poser, err1, true},
		{poser, err3, true},
		{poser, erra, false},
		{poser, errb, false},
		{errorUncomparable{}, errorUncomparable{}, true},
		{errorUncomparable{}, &errorUncomparable{}, false},
		{&errorUncomparable{}, errorUncomparable{}, true},
		{&errorUncomparable{}, &errorUncomparable{}, false},
		{errorUncomparable{}, err1, false},
		{&errorUncomparable{}, err1, false},
		{multiErr{}, err1, false},
		{multiErr{err1, err3}, err1, true},
		{multiErr{err3, err1}, err1, true},
		{multiErr{err1, err3}, errors.New("x"), false},
		{multiErr{err3, errb}, errb, true},
		{multiErr{err3, errb}, erra, true},
		{multiErr{err3, errb}, err1, true},
		{multiErr{errb, err3}, err1, true},
		{multiErr{poser}, err1, true},
		{multiErr{poser}, err3, true},
		{multiErr{nil}, nil, false},
	}
	for _, tc := range testCases {
		t.Run("", func(t *testing.T) {
			if got := errors.Is(tc.err, tc.target); got != tc.match {
				t.Errorf("Is(%v, %v) = %v, want %v", tc.err, tc.target, got, tc.match)
			}
		})
	}
}

type poser struct {
	msg string
	f   func(error) bool
}

var poserPathErr = &fs.PathError{Op: "poser"}

func (p *poser) Error() string     { return p.msg }
func (p *poser) Is(err error) bool { return p.f(err) }
func (p *poser) As(err any) bool {
	switch x := err.(type) {
	case **poser:
		*x = p
	case *errorT:
		*x = errorT{"poser"}
	case **fs.PathError:
		*x = poserPathErr
	default:
		return false
	}
	return true
}

func TestAs(t *testing.T) {
	var errT errorT
	var errP *fs.PathError
	var timeout interface{ Timeout() bool }
	var p *poser
	_, errF := os.Open("non-existing")
	poserErr := &poser{"oh no", nil}

	testCases := []struct {
		err    error
		target any
		match  bool
		want   any // value of target on match
	}{{
		nil,
		&errP,
		false,
		nil,
	}, {
		wrapped{"pitied the fool", errorT{"T"}},
		&errT,
		true,
		errorT{"T"},
	}, {
		errF,
		&errP,
		true,
		errF,
	}, {
		errorT{},
		&errP,
		false,
		nil,
	}, {
		wrapped{"wrapped", nil},
		&errT,
		false,
		nil,
	}, {
		&poser{"error", nil},
		&errT,
		true,
		errorT{"poser"},
	}, {
		&poser{"path", nil},
		&errP,
		true,
		poserPathErr,
	}, {
		poserErr,
		&p,
		true,
		poserErr,
	}, {
		errors.New("err"),
		&timeout,
		false,
		nil,
	}, {
		errF,
		&timeout,
		true,
		errF,
	}, {
		wrapped{"path error", errF},
		&timeout,
		true,
		errF,
	}, {
		multiErr{},
		&errT,
		false,
		nil,
	}, {
		multiErr{errors.New("a"), errorT{"T"}},
		&errT,
		true,
		errorT{"T"},
	}, {
		multiErr{errorT{"T"}, errors.New("a")},
		&errT,
		true,
		errorT{"T"},
	}, {
		multiErr{errorT{"a"}, errorT{"b"}},
		&errT,
		true,
		errorT{"a"},
	}, {
		multiErr{multiErr{errors.New("a"), errorT{"a"}}, errorT{"b"}},
		&errT,
		true,
		errorT{"a"},
	}, {
		multiErr{wrapped{"path error", errF}},
		&timeout,
		true,
		errF,
	}, {
		multiErr{nil},
		&errT,
		false,
		nil,
	}}
	for i, tc := range testCases {
		name := fmt.Sprintf("%d:As(Errorf(..., %v), %v)", i, tc.err, tc.target)
		// Clear the target pointer, in case it was set in a previous test.
		rtarget := reflect.ValueOf(tc.target)
		rtarget.Elem().Set(reflect.Zero(reflect.TypeOf(tc.target).Elem()))
		t.Run(name, func(t *testing.T) {
			match := errors.As(tc.err, tc.target)
			if match != tc.match {
				t.Fatalf("match: got %v; want %v", match, tc.match)
			}
			if !match {
				return
			}
			if got := rtarget.Elem().Interface(); got != tc.want {
				t.Fatalf("got %#v, want %#v", got, tc.want)
			}
		})
	}
}

func TestAsValidation(t *testing.T) {
	var s string
	testCases := []any{
		nil,
		(*int)(nil),
		"error",
		&s,
	}
	err := errors.New("error")
	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%T(%v)", tc, tc), func(t *testing.T) {
			defer func() {
				recover()
			}()
			if errors.As(err, tc) {
				t.Errorf("As(err, %T(%v)) = true, want false", tc, tc)
				return
			}
			t.Errorf("As(err, %T(%v)) did not panic", tc, tc)
		})
	}
}

func TestAsType(t *testing.T) {
	var errT errorT
	var errP *fs.PathError
	type timeout interface {
		Timeout() bool
		error
	}
	_, errF := os.Open("non-existing")
	poserErr := &poser{"oh no", nil}

	testAsType(t,
		nil,
		errP,
		false,
	)
	testAsType(t,
		wrapped{"pitied the fool", errorT{"T"}},
		errorT{"T"},
		true,
	)
	testAsType(t,
		errF,
		errF,
		true,
	)
	testAsType(t,
		errT,
		errP,
		false,
	)
	testAsType(t,
		wrapped{"wrapped", nil},
		errT,
		false,
	)
	testAsType(t,
		&poser{"error", nil},
		errorT{"poser"},
		true,
	)
	testAsType(t,
		&poser{"path", nil},
		poserPathErr,
		true,
	)
	testAsType(t,
		poserErr,
		poserErr,
		true,
	)
	testAsType(t,
		errors.New("err"),
		timeout(nil),
		false,
	)
	testAsType(t,
		errF,
		errF.(timeout),
		true)
	testAsType(t,
		wrapped{"path error", errF},
		errF.(timeout),
		true,
	)
	testAsType(t,
		multiErr{},
		errT,
		false,
	)
	testAsType(t,
		multiErr{errors.New("a"), errorT{"T"}},
		errorT{"T"},
		true,
	)
	testAsType(t,
		multiErr{errorT{"T"}, errors.New("a")},
		errorT{"T"},
		true,
	)
	testAsType(t,
		multiErr{errorT{"a"}, errorT{"b"}},
		errorT{"a"},
		true,
	)
	testAsType(t,
		multiErr{multiErr{errors.New("a"), errorT{"a"}}, errorT{"b"}},
		errorT{"a"},
		true,
	)
	testAsType(t,
		multiErr{wrapped{"path error", errF}},
		errF.(timeout),
		true,
	)
	testAsType(t,
		multiErr{nil},
		errT,
		false,
	)
}

type compError interface {
	comparable
	error
}

func testAsType[E compError](t *testing.T, err error, want E, wantOK bool) {
	t.Helper()
	name := fmt.Sprintf("AsType[%T](Errorf(..., %v))", want, err)
	t.Run(name, func(t *testing.T) {
		got, gotOK := errors.AsType[E](err)
		if gotOK != wantOK || got != want {
			t.Fatalf("got %v, %t; want %v, %t", got, gotOK, want, wantOK)
		}
	})
}

func BenchmarkIs(b *testing.B) {
	err1 := errors.New("1")
	err2 := multiErr{multiErr{multiErr{err1, errorT{"a"}}, errorT{"b"}}}

	for i := 0; i < b.N; i++ {
		if !errors.Is(err2, err1) {
			b.Fatal("Is failed")
		}
	}
}

func BenchmarkAs(b *testing.B) {
	err := multiErr{multiErr{multiErr{errors.New("a"), errorT{"a"}}, errorT{"b"}}}
	for i := 0; i < b.N; i++ {
		var target errorT
		if !errors.As(err, &target) {
			b.Fatal("As failed")
		}
	}
}

func BenchmarkAsType(b *testing.B) {
	err := multiErr{multiErr{multiErr{errors.New("a"), errorT{"a"}}, errorT{"b"}}}
	for range b.N {
		if _, ok := errors.AsType[errorT](err); !ok {
			b.Fatal("AsType failed")
		}
	}
}

func TestUnwrap(t *testing.T) {
	err1 := errors.New("1")
	erra := wrapped{"wrap 2", err1}

	testCases := []struct {
		err  error
		want error
	}{
		{nil, nil},
		{wrapped{"wrapped", nil}, nil},
		{err1, nil},
		{erra, err1},
		{wrapped{"wrap 3", erra}, erra},
	}
	for _, tc := range testCases {
		if got := errors.Unwrap(tc.err); got != tc.want {
			t.Errorf("Unwrap(%v) = %v, want %v", tc.err, got, tc.want)
		}
	}
}

type errorT struct{ s string }

func (e errorT) Error() string { return fmt.Sprintf("errorT(%s)", e.s) }

type wrapped struct {
	msg string
	err error
}

func (e wrapped) Error() string { return e.msg }
func (e wrapped) Unwrap() error { return e.err }

type multiErr []error

func (m multiErr) Error() string   { return "multiError" }
func (m multiErr) Unwrap() []error { return []error(m) }

type errorUncomparable struct {
	f []string
}

func (errorUncomparable) Error() string {
	return "uncomparable error"
}

func (errorUncomparable) Is(target error) bool {
	_, ok := target.(errorUncomparable)
	return ok
}

func TestIsAny(t *testing.T) {
	err1 := errors.New("1")
	err2 := errors.New("2")
	err3 := errors.New("3")
	erra := wrapped{"wrap a", err1}
	errb := wrapped{"wrap b", err2}

	poser := &poser{"either 1 or 3", func(err error) bool {
		return err == err1 || err == err3
	}}

	testCases := []struct {
		err     error
		targets []error
		match   bool
	}{
		// Basic cases
		{nil, []error{nil}, true},
		{nil, []error{err1}, false},
		{err1, []error{nil}, false},
		{err1, []error{err1}, true},
		{err1, []error{err2}, false},
		{err1, []error{err1, err2}, true},
		{err1, []error{err2, err1}, true},
		{err1, []error{err2, err3}, false},

		// Wrapped errors
		{erra, []error{err1}, true},
		{erra, []error{err2}, false},
		{erra, []error{err1, err2}, true},
		{erra, []error{err2, err1}, true},
		{erra, []error{err2, err3}, false},

		// Multiple targets with wrapped errors
		{errb, []error{err1, err2, err3}, true},
		{errb, []error{err1, err3}, false},

		// Posers
		{poser, []error{err1}, true},
		{poser, []error{err3}, true},
		{poser, []error{err2}, false},
		{poser, []error{err1, err2}, true},
		{poser, []error{err2, err3}, true},
		{poser, []error{err2, erra}, false},

		// Multi errors
		{multiErr{}, []error{err1}, false},
		{multiErr{err1, err2}, []error{err1}, true},
		{multiErr{err1, err2}, []error{err2}, true},
		{multiErr{err1, err2}, []error{err3}, false},
		{multiErr{err1, err2}, []error{err3, err1}, true},
		{multiErr{err1, err2}, []error{err3, erra}, false},
		{multiErr{erra, errb}, []error{err1, err2}, true},
		{multiErr{erra, errb}, []error{err3, err1}, true},

		// Empty targets
		{err1, []error{}, false},
		{nil, []error{}, false},

		// Uncomparable errors
		{errorUncomparable{}, []error{errorUncomparable{}}, true},
		{&errorUncomparable{}, []error{errorUncomparable{}}, true},
		{errorUncomparable{}, []error{err1, errorUncomparable{}}, true},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			if got := errors.IsAny(tc.err, tc.targets...); got != tc.match {
				t.Errorf("IsAny(%v, %v) = %v, want %v", tc.err, tc.targets, got, tc.match)
			}
		})
	}
}

func TestMatch(t *testing.T) {
	err1 := errors.New("1")
	err2 := errors.New("2")
	err3 := errors.New("3")
	erra := wrapped{"wrap a", err1}

	poser := &poser{"either 1 or 3", func(err error) bool {
		return err == err1 || err == err3
	}}

	testCases := []struct {
		err     error
		targets []error
		want    error // the expected matched error
	}{
		{err1, []error{err1}, err1},
		{err1, []error{err2}, nil},
		{err1, []error{err1, err2}, err1},
		{err1, []error{err2, err1}, err1}, // Returns first match (err1)
		{err1, []error{err2, err3}, nil},
		{erra, []error{err1, err2}, err1},
		{erra, []error{err2, err1}, err1}, // erra wraps err1, so matches err1
		{erra, []error{err2, err3}, nil},
		{nil, []error{nil}, nil},
		{nil, []error{err1}, nil},
		{err1, []error{}, nil},

		// Posers - note that the poser matches err1 or err3
		{poser, []error{err1}, err1},
		{poser, []error{err3}, err3},
		{poser, []error{err2}, nil},
		{poser, []error{err2, err1}, err1},
		{poser, []error{err1, err3}, err1}, // Returns first match

		// Multi errors
		{multiErr{err1, err2}, []error{err1}, err1},
		{multiErr{err1, err2}, []error{err2}, err2},
		{multiErr{err1, err2}, []error{err3}, nil},
		{multiErr{err1, err2}, []error{err3, err2}, err2},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			got := errors.Match(tc.err, tc.targets...)
			if got != tc.want {
				t.Errorf("Match(%v, %v) = %v, want %v", tc.err, tc.targets, got, tc.want)
			}
		})
	}
}

// TODO remove
func BenchmarkIsAnySlow(b *testing.B) {
	err1 := errors.New("1")
	err2 := errors.New("2")
	err3 := errors.New("3")
	err := multiErr{multiErr{multiErr{err1, errorT{"a"}}, errorT{"b"}}}

	b.Run("one_target", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !errors.IsAnySlow(err, err1) {
				b.Fatal("IsAny failed")
			}
		}
	})

	b.Run("three_targets", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !errors.IsAnySlow(err, err2, err3, err1) {
				b.Fatal("IsAny failed")
			}
		}
	})

	b.Run("no_match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if errors.IsAnySlow(err, err2, err3) {
				b.Fatal("IsAny should not match")
			}
		}
	})
}

func BenchmarkIsAny(b *testing.B) {
	err1 := errors.New("1")
	err2 := errors.New("2")
	err3 := errors.New("3")
	err := multiErr{multiErr{multiErr{err1, errorT{"a"}}, errorT{"b"}}}

	b.Run("one_target", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !errors.IsAny(err, err1) {
				b.Fatal("IsAny failed")
			}
		}
	})

	b.Run("three_targets", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if !errors.IsAny(err, err2, err3, err1) {
				b.Fatal("IsAny failed")
			}
		}
	})

	b.Run("no_match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if errors.IsAny(err, err2, err3) {
				b.Fatal("IsAny should not match")
			}
		}
	})
}

func BenchmarkMatch(b *testing.B) {
	err1 := errors.New("1")
	err2 := errors.New("2")
	err3 := errors.New("3")
	err := multiErr{multiErr{multiErr{err1, errorT{"a"}}, errorT{"b"}}}

	b.Run("one_target", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if errors.Match(err, err1) != err1 {
				b.Fatal("Match failed")
			}
		}
	})

	b.Run("three_targets", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if errors.Match(err, err2, err3, err1) != err1 {
				b.Fatal("Match failed")
			}
		}
	})

	b.Run("no_match", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if errors.Match(err, err2, err3) != nil {
				b.Fatal("Match should not match")
			}
		}
	})
}
