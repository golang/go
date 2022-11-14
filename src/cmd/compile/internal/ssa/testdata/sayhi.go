package foo

import (
	"fmt"
	"sync"
)

func sayhi(n int, wg *sync.WaitGroup) {
	fmt.Println("hi", n)
	fmt.Println("hi", n)
	wg.Done()
}
