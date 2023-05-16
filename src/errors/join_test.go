// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package errors_test

import (
	"errors"
	"reflect"
	"testing"
)

func TestJoinReturnsNil(t *testing.T) {
	if err := errors.Join(); err != nil {
		t.Errorf("errors.Join() = %v, want nil", err)
	}
	if err := errors.Join(nil); err != nil {
		t.Errorf("errors.Join(nil) = %v, want nil", err)
	}
	if err := errors.Join(nil, nil); err != nil {
		t.Errorf("errors.Join(nil, nil) = %v, want nil", err)
	}
}

func TestJoin(t *testing.T) {
	err1 := errors.New("err1")
	err2 := errors.New("err2")
	for _, test := range []struct {
		errs []error
		want []error
	}{{
		errs: []error{err1},
		want: []error{err1},
	}, {
		errs: []error{err1, err2},
		want: []error{err1, err2},
	}, {
		errs: []error{err1, nil, err2},
		want: []error{err1, err2},
	}} {
		got := errors.Join(test.errs...).(interface{ Unwrap() []error }).Unwrap()
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("Join(%v) = %v; want %v", test.errs, got, test.want)
		}
		if len(got) != cap(got) {
			t.Errorf("Join(%v) returns errors with len=%v, cap=%v; want len==cap", test.errs, len(got), cap(got))
		}
	}
}

func TestJoinErrorMethod(t *testing.T) {
	err1 := errors.New("err1")
	err2 := errors.New("err2")
	for _, test := range []struct {
		errs []error
		want string
	}{{
		errs: []error{err1},
		want: "err1",
	}, {
		errs: []error{err1, err2},
		want: "err1\nerr2",
	}, {
		errs: []error{err1, nil, err2},
		want: "err1\nerr2",
	}} {
		got := errors.Join(test.errs...).Error()
		if got != test.want {
			t.Errorf("Join(%v).Error() = %q; want %q", test.errs, got, test.want)
		}
	}
}

func TestJoinWithJoinedError(t *testing.T) {
	err1 := errors.New("err1")
	err2 := errors.New("err2")

	var err error
	err = errors.Join(err, err1)
	if err == nil {
		t.Fatal("errors.Join(err, err1) = nil, want non-nil")
	}

	gotErrs := err.(interface{ Unwrap() []error }).Unwrap()
	if len(gotErrs) != 1 {
		t.Fatalf("errors.Join(err, err1) returns errors with len=%v, want len==1", len(gotErrs))
	}

	err = errors.Join(err, err2)
	if err == nil {
		t.Fatal("errors.Join(err, err2) = nil, want non-nil")
	}

	gotErrs = err.(interface{ Unwrap() []error }).Unwrap()
	if len(gotErrs) != 2 {
		t.Fatalf("errors.Join(err, err2) returns errors with len=%v, want len==2", len(gotErrs))
	}

	// Wraps the error again, so the resulting joined error will have len==1
	err = errors.Join(err, nil)
	if err == nil {
		t.Fatal("errors.Join(err, nil) = nil, want non-nil")
	}

	gotErrs = err.(interface{ Unwrap() []error }).Unwrap()
	if len(gotErrs) != 1 {
		t.Fatalf("errors.Join(err, nil) returns errors with len=%v, want len==1", len(gotErrs))
	}

	if err.Error() != "err1\nerr2" {
		t.Errorf("Join(err, nil).Error() = %q; want %q", err.Error(), "err1\nerr2")
	}

	if _, ok := gotErrs[0].(interface{ Unwrap() []error }); !ok {
		t.Error("first error returned by errors.Join(err, nil) is not a joined error")
	}
}
