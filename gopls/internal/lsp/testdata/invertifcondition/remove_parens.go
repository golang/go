package invertifcondition

import (
	"fmt"
)

func RemoveParens() {
	b := true
	if !(b) { //@suggestedfix("if", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
