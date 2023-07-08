package extract

import "context"

type B struct {
	x int
	y int
}

func (b *B) AddP(ctx context.Context) (int, error) {
	sum := b.x + b.y
	return sum, ctx.Err() //@extractmethod("return", "ctx.Err()"),extractfunc("return", "ctx.Err()")
}

func (b *B) LongList(ctx context.Context) (int, error) {
	p1 := 1
	p2 := 1
	p3 := 1
	return p1 + p2 + p3, ctx.Err() //@extractmethod("return", "ctx.Err()"),extractfunc("return", "ctx.Err()")
}
