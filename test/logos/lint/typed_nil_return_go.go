// errorcheck

package p

import (
	"io"
	"os"
)

var touched bool

type Err struct{ msg string }

func (e *Err) Error() string { return e.msg }

func explicitError() error {
	return (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
}

func explicitParen() error {
	return ((*Err)(nil)) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
}

func explicitAny() any {
	return (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as any result is a typed nil"
}

func explicitCloser() io.Closer {
	return (*os.File)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*os.File\)\(nil\) returned as io.Closer result is a typed nil"
}

func explicitSecond() (int, error) {
	return 1, (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
}

func explicitClosure() func() error {
	return func() error {
		return (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
	}
}

func explicitUnderGuard(e *Err) error {
	if e == nil {
		return (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
	}
	return nil
}

func nilIsFine() error {
	return nil
}

func freshPointerIsFine() error {
	return &Err{msg: "x"}
}

func notAnInterface() *Err {
	return (*Err)(nil)
}

func notTheInterfaceSlot() (*Err, error) {
	return (*Err)(nil), nil
}

func otherNilConversionIsFine() any {
	return []int(nil)
}

func variableIsInactiveInGo(p *Err) error {
	return p
}

func nilVariableIsInactiveInGo() error {
	var p *Err
	return p
}

func platformPointerIsInactiveInGo() io.Closer {
	return os.Stdout
}

func namedResultIsInactiveInGo() (err error) {
	err = (*Err)(nil)
	return
}

//logos:allow(correctness::typed_nil_return)
func allowSilencesInGo() error {
	return (*Err)(nil)
}

func allowOnlyCoversItsStatement() error {
	//logos:allow(correctness::typed_nil_return)
	touched = true
	return (*Err)(nil) // ERROR "\[correctness::typed_nil_return\] \(\*Err\)\(nil\) returned as error result is a typed nil"
}
