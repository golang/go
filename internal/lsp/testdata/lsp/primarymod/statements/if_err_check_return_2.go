package statements

import "os"

func two() error {
	var s struct{ err error }

	/* if s.err != nil { return s.err } */ //@item(stmtTwoIfErrReturn, "if s.err != nil { return s.err }", "", "")

	_, s.err = os.Open("foo")
	//@snippet("", stmtTwoIfErrReturn, "", "if s.err != nil {\n\treturn ${1:s.err}\n\\}")
}
