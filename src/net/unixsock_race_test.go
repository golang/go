package net_test

import (
	"net"
	"os"
	"testing"
	"time"
)

func TestUnixListener_SetUnlinkOnClose_Concurrent(t *testing.T) {
	for i := 0; i < 100; i++ {
		sockName := "test_race.sock"
		os.Remove(sockName)
		
		l, err := net.Listen("unix", sockName)
		if err != nil {
			t.Fatal(err)
		}
		
		ul := l.(*net.UnixListener)
		
		// We create a channel to coordinate the race.
		// Actually, we'll just run them concurrently and check the final state.
		
		done := make(chan bool)
		
		// Goroutine A calls Close
		go func() {
			// small delay to let B start
			time.Sleep(10 * time.Microsecond)
			ul.Close()
			done <- true
		}()
		
		// Goroutine B calls SetUnlinkOnClose(false) then Close()
		go func() {
			ul.SetUnlinkOnClose(false)
			ul.Close()
			done <- true
		}()
		
		<-done
		<-done
		
		// After BOTH closures, the socket file should NOT be deleted,
		// because B explicitly disabled unlinking before it called Close!
		// However, due to the atomic bool + sync.Once race, if A enters unlinkOnce.Do
		// and reads true, then B disables unlink, B's Close blocks on A,
		// A unlinks the file, B's Close proceeds.
		// The file is deleted despite B's explicit disabling!
		
		if _, err := os.Stat(sockName); os.IsNotExist(err) {
			t.Fatalf("BUG: Socket file deleted explicitly after SetUnlinkOnClose(false) + Close()")
		}
		
		os.Remove(sockName)
	}
}
