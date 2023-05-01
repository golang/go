package invertifcondition

import (
	"fmt"
	"os"
)

func ElseIf() {
	// No inversion expected when there's not else clause
	if len(os.Args) > 2 {
		fmt.Println("A")
	}

	// No inversion expected for else-if, that would become unreadable
	if len(os.Args) > 2 {
		fmt.Println("A")
	} else if os.Args[0] == "X" { //@suggestedfix(re"if os.Args.0. == .X.", "refactor.rewrite", "")
		fmt.Println("B")
	} else {
		fmt.Println("C")
	}
}
