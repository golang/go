package gox

import (
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func runSource(t *testing.T, src string) (string, error) {
	t.Helper()
	r, e := TransformFile("main.gox", []byte(src), Options{BareJSX: true})
	if e != nil {
		return "", e
	}
	d := t.TempDir()
	cwd, _ := os.Getwd()
	os.WriteFile(filepath.Join(d, "go.mod"), []byte("module fixture\n\ngo 1.26\n\nrequire github.com/alfu32/go-shared/gox v0.0.0\nreplace github.com/alfu32/go-shared/gox => "+filepath.ToSlash(cwd)+"\n"), 0600)
	os.WriteFile(filepath.Join(d, "main.go"), r.Source, 0600)
	c := exec.Command(filepath.Join(runtime.GOROOT(), "bin/go"), "run", ".")
	c.Dir = d
	c.Env = append(os.Environ(), "GOWORK=off", "GOTOOLCHAIN=local")
	b, e := c.CombinedOutput()
	return string(b), e
}
func TestExecute(t *testing.T) {
	out, e := runSource(t, `package main
import("fmt"; v "github.com/alfu32/go-shared/gox/goxview")
var counter int
type S int
func(s S)String()string{counter++;return "s"}
component Counter(title string){n,set:=useState(0);return X"<button onClick={func(){set.Update(func(x int)int{return x+1})}}>{F"{title}:{n}"}</button>"}
func main(){
func fact(n int)int{if n<2{return 1};return n*fact(n-1)}
fmt.Println(F"{S(0)}{counter}",do int{return fact(5)})
r:=v.NewRoot(nil,nil);if e:=r.Mount(<Counter title="count"/>);e!=nil{panic(e)}
fmt.Println(r.Tree().Children[0].Text)
r.Tree().Attrs[0].Value.(func())();if e:=r.Flush();e!=nil{panic(e)}
fmt.Println(r.Tree().Children[0].Text)
}`)
	if e != nil {
		t.Fatalf("%v\n%s", e, out)
	}
	if out != "s1 120\ncount:0\ncount:1\n" {
		t.Fatalf("unexpected %q", out)
	}
}
func TestInvalid(t *testing.T) {
	for _, s := range []string{`package p;var x=F"{}"`, `package p;var x=X"<a></b>"`, `package p;var x=X"<a k="x" k="y"/>"`, `package p;var x=F"unclosed`, `package p;component C(){if true{n,_:=useState(0);_ =n};return X"<a/>"}`} {
		t.Run(s, func(t *testing.T) {
			if _, e := TransformFile("bad.gox", []byte(s), Options{BareJSX: true}); e == nil {
				t.Fatal("accepted invalid syntax")
			}
		})
	}
}
func TestImportShadow(t *testing.T) {
	out, e := runSource(t, `package main;import "fmt";func main(){fmt.Println(func()string{fmt:=1;return F"{fmt}"}())}`)
	if e != nil {
		t.Fatalf("%v\n%s", e, out)
	}
	if strings.TrimSpace(out) != "1" {
		t.Fatal(out)
	}
}

func TestComponentTypeChecking(t *testing.T) {
	for _, markup := range []string{`<C/>`, `<C value={1}/>`, `<C value="ok" extra="bad"/>`, `<C value="ok">child</C>`} {
		out, e := runSource(t, `package main; component C(value string){return <label>{value}</label>};func main(){_ = `+markup+`}`)
		if e == nil {
			t.Fatalf("accepted %s: %s", markup, out)
		}
	}
	out, e := runSource(t, `package main;import("fmt";v "github.com/alfu32/go-shared/gox/goxview"); component C(value *int,children []v.Node){return <label>{children}</label>};func main(){v:=1;_ =v;n:=<C value={nil}>ok</C>;fmt.Println(n.Children[0].Text)}`)
	if e != nil {
		t.Fatalf("%v\n%s", e, out)
	}
	if out != "ok\n" {
		t.Fatal(out)
	}
}

func TestLoopCaptureAndMarkupOrder(t *testing.T) {
	out, e := runSource(t, `package main
import "fmt"
func main(){
 var fs []func() int
 for i:=0;i<3;i++ {func get()int{return i};fs=append(fs,get)}
 fmt.Println(fs[0](),fs[1](),fs[2]())
 n:=0; next:=func()int{n++;return n}
 tree:=<a first={next()} second={next()}><b third={next()}/>{F"{next()}"}</a>
 fmt.Println(tree.Attrs[0].Value,tree.Attrs[1].Value,tree.Children[0].Attrs[0].Value,tree.Children[1].Text,n)
}`)
	if e != nil {
		t.Fatalf("%v\n%s", e, out)
	}
	if out != "0 1 2\n1 2 3 4 4\n" {
		t.Fatal(out)
	}
}
