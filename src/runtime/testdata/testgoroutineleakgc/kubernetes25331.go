/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/25331
 * Buggy version: 5dd087040bb13434f1ddf2f0693d0203c30f28cb
 * fix commit-id: 97f4647dc3d8cf46c2b66b89a31c758a6edfb57c
 * Flaky: 100/100
 * Description:
 *   In reflector.go, it could probably call Stop() without retrieving
 * all results from ResultChan(). See here. A potential leak is that
 * when an error has happened, it could block on resultChan, and then
 * cancelling context in Stop() wouldn't unblock it.
 */
package main

import (
	"context"
	"errors"
	"os"
	"runtime/pprof"
	"time"
)

func init() {
	register("Kubernetes25331", Kubernetes25331)
}

type watchChan_kubernetes25331 struct {
	ctx        context.Context
	cancel     context.CancelFunc
	resultChan chan bool
	errChan    chan error
}

func (wc *watchChan_kubernetes25331) Stop() {
	wc.errChan <- errors.New("Error")
	wc.cancel()
}

func (wc *watchChan_kubernetes25331) run() {
	select {
	case err := <-wc.errChan:
		errResult := len(err.Error()) != 0
		wc.cancel() // Removed in fix
		wc.resultChan <- errResult
	case <-wc.ctx.Done():
	}
}

func NewWatchChan_kubernetes25331() *watchChan_kubernetes25331 {
	ctx, cancel := context.WithCancel(context.Background())
	return &watchChan_kubernetes25331{
		ctx:        ctx,
		cancel:     cancel,
		resultChan: make(chan bool),
		errChan:    make(chan error),
	}
}

///
/// G1					G2
/// wc.run()
///						wc.Stop()
///						wc.errChan <-
///						wc.cancel()
///	<-wc.errChan
///	wc.cancel()
///	wc.resultChan <-
///	-------------G1 leak----------------
///

func Kubernetes25331() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()
	go func() {
		wc := NewWatchChan_kubernetes25331()
		// deadlocks: 1
		go wc.run()  // G1
		go wc.Stop() // G2
	}()
}
