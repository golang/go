package invertifcondition

import (
	"fmt"
)

func SemicolonAnd() {
	if n, err := fmt.Println("x"); err != nil && n > 0 { //@suggestedfix("f", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
