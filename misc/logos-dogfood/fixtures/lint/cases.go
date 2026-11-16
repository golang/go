// Package lintcases deliberately contains paired valid and invalid ownership
// and control-flow patterns. The invalid functions must never be executed.
package lintcases

type resource struct{ closed bool }

func acquire() *resource { return &resource{} }

func (r *resource) Close() error {
	r.closed = true
	return nil
}

func CloseChecked() error {
	r := acquire()
	if err := r.Close(); err != nil {
		return err
	}
	return nil
}

func CloseAssigned() error {
	r := acquire()
	err := r.Close()
	return err
}

func CloseReturned() error {
	r := acquire()
	return r.Close()
}

func Leak() bool {
	r := acquire()
	return r.closed
}

func ConditionalClose(skip bool) bool {
	r := acquire()
	return skip && r.Close() == nil
}

type failure struct{}

func (*failure) Error() string { return "failure" }

func TypedNilAfterShadowedPanic() error {
	var err *failure
	panic := func(any) {}
	panic("this function returns")
	return err
}

func RecurseAfterShadowedPanic() {
	panic := func(any) {}
	panic("this function returns")
	RecurseAfterShadowedPanic()
}

func BuiltinPanicStops() {
	(panic)("this builtin stops execution")
	BuiltinPanicStops()
}
