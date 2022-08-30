package inlayHint //@inlayHint("package")

import "fmt"

func hello(name string) string {
	return "Hello " + name
}

func helloWorld() string {
	return hello("World")
}

type foo struct{}

func (*foo) bar(baz string, qux int) int {
	if baz != "" {
		return qux + 1
	}
	return qux
}

func kase(foo int, bar bool, baz ...string) {
	fmt.Println(foo, bar, baz)
}

func kipp(foo string, bar, baz string) {
	fmt.Println(foo, bar, baz)
}

func plex(foo, bar string, baz string) {
	fmt.Println(foo, bar, baz)
}

func tars(foo string, bar, baz string) {
	fmt.Println(foo, bar, baz)
}

func foobar() {
	var x foo
	x.bar("", 1)
	kase(0, true, "c", "d", "e")
	kipp("a", "b", "c")
	plex("a", "b", "c")
	tars("a", "b", "c")
	foo, bar, baz := "a", "b", "c"
	kipp(foo, bar, baz)
	plex("a", bar, baz)
	tars(foo+foo, (bar), "c")

}
