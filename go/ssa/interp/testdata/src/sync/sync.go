package sync

// Rudimentary implementation of a mutex for interp tests.
type Mutex struct {
	c chan int // Mutex is held when held c!=nil and is empty. Access is guarded by g.
}

func (m *Mutex) Lock() {
	c := ch(m)
	<-c
}

func (m *Mutex) Unlock() {
	c := ch(m)
	c <- 1
}

// sequentializes Mutex.c access.
var g = make(chan int, 1)

func init() {
	g <- 1
}

// ch initializes the m.c field if needed and returns it.
func ch(m *Mutex) chan int {
	<-g
	defer func() {
		g <- 1
	}()
	if m.c == nil {
		m.c = make(chan int, 1)
		m.c <- 1
	}
	return m.c
}
