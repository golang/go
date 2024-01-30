package timer

// Package time knows the layout of this structure.
type Timer struct {
	// If this timer is on a heap, which P's heap it is on.
	// puintptr rather than *p to match uintptr in the versions
	// of this struct defined in other packages.
	Pp uintptr

	// Timer wakes up at when, and then at when+period, ... (period > 0 only)
	// each time calling f(arg, now) in the timer goroutine, so f must be
	// a well-behaved function and not block.
	//
	// when must be positive on an active timer.
	When   int64
	Period int64
	F      func(any, uintptr)
	Arg    any
	Seq    uintptr

	// What to set the when field to in timerModifiedXX status.
	Nextwhen int64

	// The status field holds one of the values below.
	Status uint32
}
