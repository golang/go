package stub

func main() {
	var br error = &customErr{} //@suggestedfix("&", "refactor.rewrite")
}

type customErr struct{}
