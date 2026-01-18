package myLog

type myCustomErrorType struct{}

func (myCustomErrorType) Error() string { return "my custom error" }

//go:noinline
func myCustomError() error {
	return myCustomErrorType{}
}

type myCustomWarningType struct{}

func (myCustomWarningType) Error() string { return "my custom warning" }

//go:noinline
func myCustomWarning() error {
	return myCustomWarningType{}
}
