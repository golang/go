package a

import "io"

type T interface {
	M0(_ int)
	M1(x, _ int) // _ (blank) caused crash
	M2() (x, _ int)
}

type S struct{}

func (S) M0(_ int) {}
func (S) M1(x, _ int) {}
func (S) M2() (x, _ int) { return }
func (_ S) M3() {}

// Snippet from x/tools/godoc/analysis/analysis.go.
// Offending code from #5470.
type Link interface {
	Start() int
	End() int
	Write(w io.Writer, _ int, start bool) // _ (blank) caused crash
}
