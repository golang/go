package runtime

// An errorString represents a runtime error described by a single string.
type errorString string

func (e errorString) RuntimeError() {}

func (e errorString) Error() string {
	return "runtime error: " + string(e)
}

func Breakpoint()

type Error interface {
	error
	RuntimeError()
}

func GC()
