package semantictokens //@ semantic("")

func f(x ...interface{}) {
}

func weirⰀd() {
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
