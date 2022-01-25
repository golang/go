package d

func D(i int) int {
	return i + 1
}

type Data struct {
	V int
}

func (d Data) Do() int {
	return d.V - 1
}
