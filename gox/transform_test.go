package gox

import (
	"strings"
	"testing"
)

func TestExtensions(t *testing.T) {
	cases := []string{
		`package p; func f(name string) string {return F"hello {name}"}`,
		`package p; func f() string {return F"a {F"b {2}"} {{ok}}"}`,
		`package p; func f() int {func fact(n int) int {if n<2{return 1};return n*fact(n-1)};return do int {return fact(5)}}`,
		`package p; func f(name string) any {return X"<panel title={F"hi {name}"}><label>{name}</label></panel>"}`,
		`package p; component Counter(title string){count,setCount:=useState(0);return X"<button onClick={func(){setCount(count+1)}}>{F"{title}: {count}"}</button>"}`,
		`package p; func f() any {return <label>{F"{2}"}</label>}`,
	}
	for _, src := range cases {
		t.Run(src, func(t *testing.T) {
			r, err := TransformFile("test.gox", []byte(src), Options{BareJSX: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Log(string(r.Source))
			if strings.Contains(string(r.Source), `F"`) {
				t.Fatal("unlowered F")
			}
		})
	}
}
