package main

type Node struct {
	Circular bool
}

type ExtNode[V any] struct {
	v V
	Node
}

type List[V any] struct {
	root *ExtNode[V]
	len  int
}

func (list *List[V]) PushBack(arg V) {
	if list.len == 0 {
		list.root = &ExtNode[V]{v: arg}
		list.root.Circular = true
		list.len++
		return
	}
	list.len++
}

func main() {
	var v List[int]
	v.PushBack(1)
}
