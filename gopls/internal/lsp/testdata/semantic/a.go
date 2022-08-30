package semantictokens //@ semantic("")

import (
	_ "encoding/utf8"
	utf "encoding/utf8"
	"fmt" //@ semantic("fmt")
	. "fmt"
	"unicode/utf8"
)

var (
	a           = fmt.Print
	b  []string = []string{"foo"}
	c1 chan int
	c2 <-chan int
	c3 = make([]chan<- int)
	b  = A{X: 23}
	m  map[bool][3]*float64
)

const (
	xx F = iota
	yy   = xx + 3
	zz   = ""
	ww   = "not " + zz
)

type A struct {
	X int `foof`
}
type B interface {
	A
	sad(int) bool
}

type F int

func (a *A) f() bool {
	var z string
	x := "foo"
	a(x)
	y := "bar" + x
	switch z {
	case "xx":
	default:
	}
	select {
	case z := <-c3[0]:
	default:
	}
	for k, v := range m {
		return (!k) && v[0] == nil
	}
	c2 <- A.X
	w := b[4:]
	j := len(x)
	j--
	q := []interface{}{j, 23i, &y}
	g(q...)
	return true
}

func g(vv ...interface{}) {
	ff := func() {}
	defer ff()
	go utf.RuneCount("")
	go utf8.RuneCount(vv.(string))
	if true {
	} else {
	}
Never:
	for i := 0; i < 10; {
		break Never
	}
	_, ok := vv[0].(A)
	if !ok {
		switch x := vv[0].(type) {
		}
		goto Never
	}
}
