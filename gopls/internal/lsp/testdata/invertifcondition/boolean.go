package invertifcondition

import (
	"fmt"
)

func Boolean() {
	b := true
	if b { //@suggestedfix("if b", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
