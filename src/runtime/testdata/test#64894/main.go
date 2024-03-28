package main

import (
	"sync"
)

func ioWaitGoroutine() {
	println("Patched select no cases with: runtime/select.go L103 with waitReasonIOWait")
	select{}
	//for {
	//	fmt.Println("Goroutine is in I/O Wait mode")
	//	time.Sleep(1 * time.Second) // Simulate I/O wait by sleeping for 1 second
	//}
}


func main() {
	var wg sync.WaitGroup
	wg.Add(100)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			ioWaitGoroutine()
		}()
	}
	wg.Wait()
}
