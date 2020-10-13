package semantictokens //@ semantic("")

func weirⰀd() {
	const (
		snil  = nil
		nil   = true
		true  = false
		false = snil
	)
}
