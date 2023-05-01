package invertifcondition

import (
	"fmt"
)

func Semicolon() {
	if _, err := fmt.Println("x"); err != nil { //@suggestedfix("if", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
