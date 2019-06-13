package time

func (t *Ticker) Reset(d Duration) bool {
	if t.r.f == nil {
		panic("time: Reset called on uninitialized Ticker")
	}
	w := when(d)
	active := stopTimer(&t.r)
	t.r.when = w
	t.r.period = int64(d)
	startTimer(&t.r)
	return active
}
