package invertifcondition

import (
	"fmt"
	"os"
)

func BooleanFn() {
	if os.IsPathSeparator('X') { //@suggestedfix("if os.IsPathSeparator('X')", "refactor.rewrite", "")
		fmt.Println("A")
	} else {
		fmt.Println("B")
	}
}
