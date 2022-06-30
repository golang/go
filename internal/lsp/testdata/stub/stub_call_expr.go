package stub

func main() {
	check(&callExpr{}) //@suggestedfix("&", "refactor.rewrite", "")
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}

type callExpr struct{}
