package myLog

type myCustomErrorType struct{}

func (myCustomErrorType) Error() string { return "my custom error" }

//go:noinline
func myCustomError() error {
	return myCustomErrorType{}
}
