package analyzer

import "testing"

func Testbad(t *testing.T) { //@diag("", "tests", "Testbad has malformed name: first letter after 'Test' must not be lowercase")

}
