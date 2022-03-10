// This interpreter test is designed to test the test copy of DeepEqual.
//
// Validate this file with 'go run' after editing.

package main

import "reflect"

func assert(cond bool) {
	if !cond {
		panic("failed")
	}
}

type X int
type Y struct {
	y *Y
	z [3]int
}

var (
	a = []int{0, 1, 2, 3}
	b = []X{0, 1, 2, 3}
	c = map[int]string{0: "zero", 1: "one"}
	d = map[X]string{0: "zero", 1: "one"}
	e = &Y{}
	f = (*Y)(nil)
	g = &Y{y: e}
	h *Y
)

func init() {
	h = &Y{} // h->h
	h.y = h
}

func main() {
	assert(reflect.DeepEqual(nil, nil))
	assert(reflect.DeepEqual((*int)(nil), (*int)(nil)))
	assert(!reflect.DeepEqual(nil, (*int)(nil)))

	assert(reflect.DeepEqual(0, 0))
	assert(!reflect.DeepEqual(0, int64(0)))

	assert(!reflect.DeepEqual("", 0))

	assert(reflect.DeepEqual(a, []int{0, 1, 2, 3}))
	assert(!reflect.DeepEqual(a, []int{0, 1, 2}))
	assert(!reflect.DeepEqual(a, []int{0, 1, 0, 3}))

	assert(reflect.DeepEqual(b, []X{0, 1, 2, 3}))
	assert(!reflect.DeepEqual(b, []X{0, 1, 0, 3}))

	assert(reflect.DeepEqual(c, map[int]string{0: "zero", 1: "one"}))
	assert(!reflect.DeepEqual(c, map[int]string{0: "zero", 1: "one", 2: "two"}))
	assert(!reflect.DeepEqual(c, map[int]string{1: "one", 2: "two"}))
	assert(!reflect.DeepEqual(c, map[int]string{1: "one"}))

	assert(reflect.DeepEqual(d, map[X]string{0: "zero", 1: "one"}))
	assert(!reflect.DeepEqual(d, map[int]string{0: "zero", 1: "one"}))

	assert(reflect.DeepEqual(e, &Y{}))
	assert(reflect.DeepEqual(e, &Y{z: [3]int{0, 0, 0}}))
	assert(!reflect.DeepEqual(e, &Y{z: [3]int{0, 1, 0}}))

	assert(reflect.DeepEqual(f, (*Y)(nil)))
	assert(!reflect.DeepEqual(f, nil))

	// eq_h -> eq_h. Pointer structure and elements are equal so DeepEqual.
	eq_h := &Y{}
	eq_h.y = eq_h
	assert(reflect.DeepEqual(h, eq_h))

	// deepeq_h->h->h. Pointed to elem of (deepeq_h, h) are (h,h). (h,h) are deep equal so h and deepeq_h are DeepEqual.
	deepeq_h := &Y{}
	deepeq_h.y = h
	assert(reflect.DeepEqual(h, deepeq_h))

	distinct := []interface{}{a, b, c, d, e, f, g, h}
	for x := range distinct {
		for y := range distinct {
			assert((x == y) == reflect.DeepEqual(distinct[x], distinct[y]))
		}
	}

	// anonymous struct types.
	assert(reflect.DeepEqual(struct{}{}, struct{}{}))
	assert(reflect.DeepEqual(struct{ x int }{1}, struct{ x int }{1}))
	assert(!reflect.DeepEqual(struct{ x int }{}, struct{ x int }{5}))
	assert(!reflect.DeepEqual(struct{ x, y int }{0, 1}, struct{ x int }{0}))
	assert(reflect.DeepEqual(struct{ x, y int }{2, 3}, struct{ x, y int }{2, 3}))
	assert(!reflect.DeepEqual(struct{ x, y int }{4, 5}, struct{ x, y int }{4, 6}))
}
