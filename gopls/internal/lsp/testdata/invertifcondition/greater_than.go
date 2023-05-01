package invertifcondition

import (
	"fmt"
	"os"
)

func GreaterThan() {
	if len(os.Args) > 2 { //@suggestedfix("i", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
