//go:build ignore
// +build ignore

package main

var a int

type t struct {
	a *map[string]chan *int
}

func fn() []t {
	m := make(map[string]chan *int)
	m[""] = make(chan *int, 1)
	m[""] <- &a
	return []t{t{a: &m}}
}

func main() {
	x := fn()
	print(x) // @pointstoquery <-(*x[i].a)[key] command-line-arguments.a
}
