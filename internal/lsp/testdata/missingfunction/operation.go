package missingfunction

import "time"

func operation() {
	undefinedOperation(10 * time.Second) //@suggestedfix("undefinedOperation", "quickfix", "")
}
