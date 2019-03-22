package errors

func New(text string) error { return errorString{text} }

type errorString struct{ s string }

func (e errorString) Error() string { return e.s }
