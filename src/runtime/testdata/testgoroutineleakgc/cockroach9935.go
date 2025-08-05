/*
 * Project: cockroach
 * Issue or PR  : https://github.com/cockroachdb/cockroach/pull/9935
 * Buggy version: 4df302cc3f03328395dc3fefbfba58b7718e4f2f
 * fix commit-id: ed6a100ba38dd51b0888b9a3d3ac6bdbb26c528c
 * Flaky: 100/100
 * Description: This bug is caused by acquiring l.mu.Lock() twice. The fix is
 * to release l.mu.Lock() before acquiring l.mu.Lock for the second time.
 */
package main

import (
	"errors"
	"math/rand"
	"runtime"
	"sync"
	"time"
)

func init() {
	register("Cockroach9935", Cockroach9935)
}

type loggingT_cockroach9935 struct {
	mu sync.Mutex
}

func (l *loggingT_cockroach9935) outputLogEntry() {
	l.mu.Lock()
	if err := l.createFile(); err != nil {
		l.exit(err)
	}
	l.mu.Unlock()
}
func (l *loggingT_cockroach9935) createFile() error {
	if rand.Intn(8)%4 > 0 {
		return errors.New("")
	}
	return nil
}
func (l *loggingT_cockroach9935) exit(err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
}
func Cockroach9935() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()

	for i := 0; i < 100; i++ {
		go func() {
			l := &loggingT_cockroach9935{}
			// deadlocks: x > 0
			go l.outputLogEntry()
		}()
	}
}
