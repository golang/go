package semantictokens //@ semantic("")

func f(x ...interface{}) {
}

func weirⰀd() { /*😀*/ // comment
	const (
		snil   = nil
		nil    = true
		true   = false
		false  = snil
		cmd    = `foof`
		double = iota
		iota   = copy
		four   = (len(cmd)/2 < 5)
		five   = four
	)
	f(cmd, nil, double, iota)
}

/*

multiline */ /*
multiline
*/
type AA int
type BB struct {
	AA
}
type CC struct {
	AA int
}
type D func(aa AA) (BB error)
type E func(AA) BB
