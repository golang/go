package statements

import (
	"bytes"
	"io"
	"os"
)

func one() (int, float32, io.Writer, *int, []int, bytes.Buffer, error) {
	/* if err != nil { return err } */ //@item(stmtOneIfErrReturn, "if err != nil { return err }", "", "")
	/* err != nil { return err } */ //@item(stmtOneErrReturn, "err != nil { return err }", "", "")

	_, err := os.Open("foo")
	//@snippet("", stmtOneIfErrReturn, "", "if err != nil {\n\treturn 0, 0, nil, nil, nil, bytes.Buffer{\\}, ${1:err}\n\\}")

	_, err = os.Open("foo")
	i //@snippet(" //", stmtOneIfErrReturn, "", "if err != nil {\n\treturn 0, 0, nil, nil, nil, bytes.Buffer{\\}, ${1:err}\n\\}")

	_, err = os.Open("foo")
	if er //@snippet(" //", stmtOneErrReturn, "", "err != nil {\n\treturn 0, 0, nil, nil, nil, bytes.Buffer{\\}, ${1:err}\n\\}")

	_, err = os.Open("foo")
	if //@snippet(" //", stmtOneIfErrReturn, "", "if err != nil {\n\treturn 0, 0, nil, nil, nil, bytes.Buffer{\\}, ${1:err}\n\\}")

	_, err = os.Open("foo")
	if //@snippet("//", stmtOneIfErrReturn, "", "if err != nil {\n\treturn 0, 0, nil, nil, nil, bytes.Buffer{\\}, ${1:err}\n\\}")
}
