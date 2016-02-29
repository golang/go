package template

import (
	"bytes"
	"testing"
)

func TestTemplateClone(t *testing.T) {
	// https://golang.org/issue/12996
	orig := New("name")
	clone, err := orig.Clone()
	if err != nil {
		t.Fatal(err)
	}
	if len(clone.Templates()) != len(orig.Templates()) {
		t.Fatalf("Invalid length of t.Clone().Templates()")
	}

	const want = "stuff"
	parsed := Must(clone.Parse(want))
	var buf bytes.Buffer
	err = parsed.Execute(&buf, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := buf.String(); got != want {
		t.Fatalf("got %q; want %q", got, want)
	}
}
