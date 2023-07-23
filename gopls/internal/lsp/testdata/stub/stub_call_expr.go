package stub

func main() {
	check(&callExpr{}) //@suggestedfix("&", "quickfix", "")
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

type callExpr struct{}
