package b

import "fmt"

// Wrapf is a printf wrapper.
func Wrapf(format string, args ...interface{}) { // want Wrapf:"printfWrapper"
	fmt.Sprintf(format, args...)
}

// Wrap is a print wrapper.
func Wrap(args ...interface{}) { // want Wrap:"printWrapper"
	fmt.Sprint(args...)
}

// NoWrap is not a wrapper.
func NoWrap(format string, args ...interface{}) {
}

// Wrapf2 is another printf wrapper.
func Wrapf2(format string, args ...interface{}) string { // want Wrapf2:"printfWrapper"

	// This statement serves as an assertion that this function is a
	// printf wrapper and that calls to it should be checked
	// accordingly, even though the delegation below is obscured by
	// the "("+format+")" operations.
	if false {
		fmt.Sprintf(format, args...)
	}

	// Effectively a printf delegation,
	// but the printf checker can't see it.
	return fmt.Sprintf("("+format+")", args...)
}
