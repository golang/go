package extract

type A struct {
	x int
	y int
}

func (a *A) XLessThanYP() bool {
	return a.x < a.y //@extractmethod("return", "a.y"),extractfunc("return", "a.y")
}

func (a *A) AddP() int {
	sum := a.x + a.y //@extractmethod("sum", "a.y"),extractfunc("sum", "a.y")
	return sum       //@extractmethod("return", "sum"),extractfunc("return", "sum")
}

func (a A) XLessThanY() bool {
	return a.x < a.y //@extractmethod("return", "a.y"),extractfunc("return", "a.y")
}

func (a A) Add() int {
	sum := a.x + a.y //@extractmethod("sum", "a.y"),extractfunc("sum", "a.y")
	return sum       //@extractmethod("return", "sum"),extractfunc("return", "sum")
}
