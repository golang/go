package context

type Context interface {
	Done() <-chan struct{}
}

func Background() Context
