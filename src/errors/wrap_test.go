// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"errors"
	"fmt"
	"os"
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

func (p *poser) Error() string     { return p.msg }
func (p *poser) Is(err error) bool { return p.f(err) }
func (p *poser) As(err interface{}) bool {
	switch x := err.(type) {
	case **poser:
		*x = p
	case *errorT:
		*x = errorT{}
	case **os.PathError:
		*x = &os.PathError{}
	default:
		return false
	}
	return true
}

func TestAs(t *testing.T) {
	var errT errorT
	var errP *os.PathError
	var timeout interface{ Timeout() bool }
	var p *poser
	_, errF := os.Open("non-existing")

	testCases := []struct {
		err    error
		target interface{}
		match  bool
	}{{
		nil,
		&errP,
		false,
	}, {
		wrapped{"pittied the fool", errorT{}},
		&errT,
		true,
	}, {
		errF,
		&errP,
		true,
	}, {
		errorT{},
		&errP,
		false,
	}, {
		wrapped{"wrapped", nil},
		&errT,
		false,
	}, {
		&poser{"error", nil},
		&errT,
		true,
	}, {
		&poser{"path", nil},
		&errP,
		true,
	}, {
		&poser{"oh no", nil},
		&p,
		true,
	}, {
		errors.New("err"),
		&timeout,
		false,
	}, {
		errF,
		&timeout,
		true,
	}, {
		wrapped{"path error", errF},
		&timeout,
		true,
	}}
	for i, tc := range testCases {
		name := fmt.Sprintf("%d:As(Errorf(..., %v), %v)", i, tc.err, tc.target)
		t.Run(name, func(t *testing.T) {
			match := errors.As(tc.err, tc.target)
			if match != tc.match {
				t.Fatalf("match: got %v; want %v", match, tc.match)
			}
			if !match {
				return
			}
			if tc.target == nil {
				t.Fatalf("non-nil result after match")
			}
		})
	}
}

func TestAsValidation(t *testing.T) {
	var s string
	testCases := []interface{}{
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

type errorT struct{}

func (errorT) Error() string { return "errorT" }

type wrapped struct {
	msg string
	err error
}

func (e wrapped) Error() string { return e.msg }

func (e wrapped) Unwrap() error { return e.err }

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
