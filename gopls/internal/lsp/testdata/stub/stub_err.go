package stub

func main() {
	var br error = &customErr{} //@suggestedfix("&", "quickfix", "")
}

type customErr struct{}
