package main

type errType string

const constErr errType = "blah"

func (et errType) Error() string {
	return string(et)
}

var errVar error = errType("foo")

func genErr(i int) error {
	switch i {
	case 0:
		return constErr
	case 1:
		return errVar
	default:
		return nil
	}
}

func unreachable() {
	err := errVar // @whicherrs func-dead "err"
	_ = err
}

func main() {
	err := genErr(0) // @whicherrs localerrs "err"
	_ = err
}
