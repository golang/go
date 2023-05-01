package invertifcondition

import (
	"fmt"
)

func SemicolonOr() {
	if n, err := fmt.Println("x"); err != nil || n < 5 { //@suggestedfix(re"if n, err := fmt.Println..x..; err != nil .. n < 5", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
