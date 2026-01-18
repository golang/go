package composite

import (
	"bytes"
	"testing"
)

type inner struct {
	n    uint16
	data []byte
}

type outer struct {
	ok    bool
	p     *inner
	tag   [3]byte
	items []inner
}

func FuzzComposite(f *testing.F) {
	f.Add(outer{
		ok:    true,
		p:     &inner{n: 31337, data: []byte("libafl")},
		tag:   [3]byte{'A', 'B', 'C'},
		items: []inner{{n: 1, data: []byte("x")}},
	})
	f.Fuzz(func(t *testing.T, in outer) {
		if in.ok &&
			in.p != nil &&
			in.p.n == 31337 &&
			bytes.Equal(in.p.data, []byte("libafl")) &&
			in.tag == [3]byte{'A', 'B', 'C'} &&
			len(in.items) == 1 &&
			in.items[0].n == 1 &&
			bytes.Equal(in.items[0].data, []byte("x")) {
			t.Fatalf("boom")
		}
	})
}
