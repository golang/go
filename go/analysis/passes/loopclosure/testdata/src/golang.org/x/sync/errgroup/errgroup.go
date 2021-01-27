// Package errgroup synthesizes Go's package "golang.org/x/sync/errgroup",
// which is used in unit-testing.
package errgroup

type Group struct {
}

func (g *Group) Go(f func() error) {
	go func() {
		f()
	}()
}
