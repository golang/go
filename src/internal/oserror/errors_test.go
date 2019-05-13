package oserror_test

import (
	"errors"
	"fmt"
	"internal/oserror"
	"os"
	"testing"
)

type ttError struct {
	timeout   bool
	temporary bool
}

func (e ttError) Error() string {
	return fmt.Sprintf("ttError{timeout:%v temporary:%v}", e.timeout, e.temporary)
}
func (e ttError) Timeout() bool   { return e.timeout }
func (e ttError) Temporary() bool { return e.temporary }

type isError struct {
	err error
}

func (e isError) Error() string        { return fmt.Sprintf("isError(%v)", e.err) }
func (e isError) Is(target error) bool { return e.err == target }

func TestIsTimeout(t *testing.T) {
	for _, test := range []struct {
		want bool
		err  error
	}{
		{true, ttError{timeout: true}},
		{true, isError{os.ErrTimeout}},
		{true, os.ErrTimeout},
		{true, fmt.Errorf("wrap: %w", os.ErrTimeout)},
		{false, ttError{timeout: false}},
		{false, errors.New("error")},
	} {
		if got, want := oserror.IsTimeout(test.err), test.want; got != want {
			t.Errorf("IsTimeout(err) = %v, want %v\n%+v", got, want, test.err)
		}
	}
}

func TestIsTemporary(t *testing.T) {
	for _, test := range []struct {
		want bool
		err  error
	}{
		{true, ttError{temporary: true}},
		{true, isError{os.ErrTemporary}},
		{true, os.ErrTemporary},
		{true, fmt.Errorf("wrap: %w", os.ErrTemporary)},
		{false, ttError{temporary: false}},
		{false, errors.New("error")},
	} {
		if got, want := oserror.IsTemporary(test.err), test.want; got != want {
			t.Errorf("IsTemporary(err) = %v, want %v\n%+v", got, want, test.err)
		}
	}
}
