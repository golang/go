package goxview

import (
	"fmt"
	"testing"
)

func TestStateIdentityAndLifetime(t *testing.T) {
	setters := map[string]StateSetter[int]{}
	refs := map[string]*Ref[int]{}
	renders := 0
	d := Define("counter", func(i *Instance, a []Attr, c []Node) Node {
		name := Prop[string](a, "name")
		n, set := UseState(i, "n", 0)
		ref := UseRef(i, "r", 0)
		if old := refs[name]; old != nil && old != ref {
			t.Fatal("lost ref")
		}
		refs[name] = ref
		setters[name] = set
		renders++
		return Text(fmt.Sprintf("%s:%d", name, n))
	})
	tree := func(names ...string) Node {
		var cs []Node
		for _, s := range names {
			cs = append(cs, Component(d, []Attr{{"key", s}, {"name", s}}, nil))
		}
		return Element("list", nil, cs)
	}
	r := NewRoot(nil, nil)
	if err := r.Mount(tree("a", "b")); err != nil {
		t.Fatal(err)
	}
	setters["a"](2)
	setters["a"].Update(func(n int) int { return n + 3 })
	if err := r.Flush(); err != nil {
		t.Fatal(err)
	}
	if r.Tree().Children[0].Text != "a:5" || r.Tree().Children[1].Text != "b:0" {
		t.Fatal(r.Tree())
	}
	before := renders
	refs["a"].Current = 9
	r.Flush()
	if renders != before {
		t.Fatal("ref scheduled render")
	}
	if err := r.Mount(tree("b", "a")); err != nil {
		t.Fatal(err)
	}
	if r.Tree().Children[1].Text != "a:5" {
		t.Fatal("keyed reorder lost state")
	}
	stale := setters["a"]
	r.Unmount()
	stale(100)
	if err := r.Flush(); err != nil {
		t.Fatal(err)
	}
}
func TestContextAndPanic(t *testing.T) {
	ctx := NewContext("default")
	bad := false
	d := Define("consumer", func(i *Instance, a []Attr, c []Node) Node {
		if bad {
			panic("broken")
		}
		return Text(UseContext(i, ctx))
	})
	r := NewRoot(nil, nil)
	if err := r.Mount(ctx.Provider("outer", ctx.Provider("inner", Component(d, nil, nil)))); err != nil {
		t.Fatal(err)
	}
	if r.Tree().Children[0].Children[0].Text != "inner" {
		t.Fatal("provider resolution")
	}
	bad = true
	if err := r.Mount(Component(d, nil, nil)); err == nil {
		t.Fatal("expected error")
	}
	if r.Tree().Tag != "#provider" {
		t.Fatal("failed render committed")
	}
	bad = false
	if err := r.Mount(Component(d, nil, nil)); err != nil {
		t.Fatal(err)
	}
	if r.Tree().Text != "default" {
		t.Fatal("default")
	}
}
func TestDuplicateKeys(t *testing.T) {
	r := NewRoot(nil, nil)
	err := r.Mount(Element("list", nil, []Node{Element("a", []Attr{{"key", "same"}}, nil), Element("b", []Attr{{"key", "same"}}, nil)}))
	if err == nil {
		t.Fatal("accepted duplicate keys")
	}
}

func TestWorkerUpdatesDuringRender(t *testing.T) {
	var set StateSetter[int]
	started, done := make(chan struct{}), make(chan struct{})
	block := false
	d := Define("worker", func(i *Instance, _ []Attr, _ []Node) Node {
		n, s := UseState(i, "n", 0)
		set = s
		if block {
			close(started)
			<-done
		}
		return Text(fmt.Sprint(n))
	})
	r := NewRoot(nil, nil)
	if e := r.Mount(Component(d, nil, nil)); e != nil {
		t.Fatal(e)
	}
	workerSet := set
	block = true
	go func() {
		<-started
		for j := 0; j < 100; j++ {
			workerSet.EnqueueUpdate(func(n int) int { return n + 1 })
		}
		close(done)
	}()
	if e := r.Mount(Component(d, nil, nil)); e != nil {
		t.Fatal(e)
	}
	block = false
	if e := r.Flush(); e != nil {
		t.Fatal(e)
	}
	if r.Tree().Text != "100" {
		t.Fatal(r.Tree())
	}
}
func TestRootKeyResetAndNilContext(t *testing.T) {
	var set StateSetter[int]
	c := NewContext[any]("default")
	d := Define("root", func(i *Instance, _ []Attr, _ []Node) Node {
		n, s := UseState(i, "n", 0)
		set = s
		return Text(fmt.Sprint(n, UseContext(i, c)))
	})
	r := NewRoot(nil, nil)
	if e := r.Mount(Component(d, []Attr{{"key", "a"}}, nil)); e != nil {
		t.Fatal(e)
	}
	set(5)
	r.Flush()
	if e := r.Mount(Component(d, []Attr{{"key", "b"}}, nil)); e != nil {
		t.Fatal(e)
	}
	if r.Tree().Text != "0default" {
		t.Fatal(r.Tree())
	}
	if e := r.Mount(c.Provider(nil, Component(d, nil, nil))); e != nil {
		t.Fatal(e)
	}
	if r.Tree().Children[0].Text != "0 <nil>" {
		t.Fatal(r.Tree())
	}
}

func TestUpdaterPanicAndRenderSetter(t *testing.T) {
	var set StateSetter[int]
	illegal := false
	d := Define("panic", func(i *Instance, _ []Attr, _ []Node) Node {
		n, s := UseState(i, "n", 0)
		set = s
		if illegal {
			s(5)
		}
		return Text(fmt.Sprint(n))
	})
	r := NewRoot(nil, nil)
	if e := r.Mount(Component(d, nil, nil)); e != nil {
		t.Fatal(e)
	}
	set.Update(func(int) int { panic("bad updater") })
	set(3)
	if e := r.Flush(); e == nil {
		t.Fatal("updater panic not reported")
	}
	if e := r.Flush(); e != nil {
		t.Fatal(e)
	}
	if r.Tree().Text != "3" {
		t.Fatal("lost remaining queue")
	}
	illegal = true
	if e := r.Mount(Component(d, nil, nil)); e == nil {
		t.Fatal("render setter accepted")
	}
	if r.Tree().Text != "3" {
		t.Fatal("render panic committed")
	}
}
