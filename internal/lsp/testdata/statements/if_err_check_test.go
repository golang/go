package statements

import (
	"os"
	"testing"
)

func TestErr(t *testing.T) {
	/* if err != nil { t.Fatal(err) } */ //@item(stmtOneIfErrTFatal, "if err != nil { t.Fatal(err) }", "", "")

	_, err := os.Open("foo")
	//@snippet("", stmtOneIfErrTFatal, "", "if err != nil {\n\tt.Fatal(err)\n\\}")
}

func BenchmarkErr(b *testing.B) {
	/* if err != nil { b.Fatal(err) } */ //@item(stmtOneIfErrBFatal, "if err != nil { b.Fatal(err) }", "", "")

	_, err := os.Open("foo")
	//@snippet("", stmtOneIfErrBFatal, "", "if err != nil {\n\tb.Fatal(err)\n\\}")
}
