package invertifcondition

import (
	"fmt"
)

func RemoveElse() {
	if true { //@suggestedfix("if true", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
		return
	}

	fmt.Println("C")
}
