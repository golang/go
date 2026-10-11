package gox

import "testing"

func BenchmarkTransform(b *testing.B) {
	src := []byte(`package p;component C(title string){n,set:=useState(0);return <button onClick={func(){set(n+1)}}>{F"{title}: {n}"}</button>}`)
	b.ReportAllocs()
	b.SetBytes(int64(len(src)))
	for b.Loop() {
		if _, e := TransformFile("benchmark.gox", src, Options{BareJSX: true}); e != nil {
			b.Fatal(e)
		}
	}
}
