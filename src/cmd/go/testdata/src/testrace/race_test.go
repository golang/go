package testrace

import "testing"

func TestRace(t *testing.T) {
	for i := 0; i < 10; i++ {
		c := make(chan int)
		x := 1
		go func() {
			x = 2
			c <- 1
		}()
		x = 3
		<-c
		_ = x
	}
}

func BenchmarkRace(b *testing.B) {
	for i := 0; i < b.N; i++ {
		c := make(chan int)
		x := 1
		go func() {
			x = 2
			c <- 1
		}()
		x = 3
		<-c
		_ = x
	}
}
