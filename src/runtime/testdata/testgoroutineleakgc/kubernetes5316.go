/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/5316
 * Buggy version: c868b0bbf09128960bc7c4ada1a77347a464d876
 * fix commit-id: cc3a433a7abc89d2f766d4c87eaae9448e3dc091
 * Flaky: 100/100
 * Description:
 *   If the main goroutine selects a case that doesn’t consumes
 * the channels, the anonymous goroutine will be blocked on sending
 * to channel.
 */

package main

import (
	"errors"
	"math/rand"
	"runtime"
	"time"
)

func init() {
	register("Kubernetes5316", Kubernetes5316)
}

func finishRequest_kubernetes5316(timeout time.Duration, fn func() error) {
	ch := make(chan bool)
	errCh := make(chan error)
	go func() { // G2
		// deadlocks: 1
		if err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- true
		}
	}()

	select {
	case <-ch:
	case <-errCh:
	case <-time.After(timeout):
	}
}

///
/// G1 						G2
/// finishRequest()
/// 						fn()
/// time.After()
/// 						errCh<-/ch<-
/// --------------G2 leak----------------
///

func Kubernetes5316() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()
	go func() {
		fn := func() error {
			time.Sleep(2 * time.Millisecond)
			if rand.Intn(10) > 5 {
				return errors.New("Error")
			}
			return nil
		}
		go finishRequest_kubernetes5316(time.Millisecond, fn) // G1
	}()
}
