package invertifcondition

import (
	"fmt"
)

func DontRemoveParens() {
	a := false
	b := true
	if !(a ||
		b) { //@suggestedfix("b", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
