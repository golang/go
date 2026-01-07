package multiargs

import (
	"bytes"
	"testing"
)

type Input struct {
	Data []byte
	S    string
	N    int
	Tag  [2]byte
	OK   bool
}

func FuzzMultiArgs(f *testing.F) {
	f.Add(Input{Data: []byte("A"), S: "B", N: 7, Tag: [2]byte{'C', 'D'}, OK: true})
	f.Fuzz(func(t *testing.T, in Input) {
		if in.OK && in.N == 7 && in.S == "B" && in.Tag == [2]byte{'C', 'D'} && bytes.Equal(in.Data, []byte("A")) {
			t.Fatalf("boom")
		}
	})
}
