// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

/*
 * Project: moby
 * Issue or PR  : https://github.com/moby/moby/pull/27782
 * Buggy version: 18768fdc2e76ec6c600c8ab57d2d487ee7877794
 * fix commit-id: a69a59ffc7e3d028a72d1195c2c1535f447eaa84
 * Flaky: 2/100
 */
package main

import (
	"errors"
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Moby27782", Moby27782)
}

type Event_moby27782 struct {
	Op Op_moby27782
}

type Op_moby27782 uint32

const (
	Create_moby27782 Op_moby27782 = 1 << iota
	Write_moby27782
	Remove_moby27782
	Rename_moby27782
	Chmod_moby27782
)

func newEvent(op Op_moby27782) Event_moby27782 {
	return Event_moby27782{op}
}

func (e *Event_moby27782) ignoreLinux(w *Watcher_moby27782) bool {
	if e.Op != Write_moby27782 {
		w.mu.Lock()
		defer w.mu.Unlock()
		w.cv.Broadcast()
		return true
	}
	runtime.Gosched()
	return false
}

type Watcher_moby27782 struct {
	Events chan Event_moby27782
	mu     sync.Mutex // L1
	cv     *sync.Cond // C1
	done   chan struct{}
}

func NewWatcher_moby27782() *Watcher_moby27782 {
	w := &Watcher_moby27782{
		Events: make(chan Event_moby27782),
		done:   make(chan struct{}),
	}
	w.cv = sync.NewCond(&w.mu)
	go w.readEvents() // G3
	return w
}

func (w *Watcher_moby27782) readEvents() {
	defer close(w.Events)
	for {
		if w.isClosed() {
			return
		}
		event := newEvent(Write_moby27782) // MODIFY event
		if !event.ignoreLinux(w) {
			runtime.Gosched()
			select {
			case w.Events <- event:
			case <-w.done:
				return
			}
		}
	}
}

func (w *Watcher_moby27782) isClosed() bool {
	select {
	case <-w.done:
		return true
	default:
		return false
	}
}

func (w *Watcher_moby27782) Close() {
	if w.isClosed() {
		return
	}
	close(w.done)
}

func (w *Watcher_moby27782) Remove() {
	w.mu.Lock()
	defer w.mu.Unlock()
	exists := true
	for exists {
		w.cv.Wait()
		runtime.Gosched()
	}
}

type FileWatcher_moby27782 interface {
	Events() <-chan Event_moby27782
	Remove()
	Close()
}

func New_moby27782() FileWatcher_moby27782 {
	return NewEventWatcher_moby27782()
}

func NewEventWatcher_moby27782() FileWatcher_moby27782 {
	return &fsNotifyWatcher_moby27782{NewWatcher_moby27782()}
}

type fsNotifyWatcher_moby27782 struct {
	*Watcher_moby27782
}

func (w *fsNotifyWatcher_moby27782) Events() <-chan Event_moby27782 {
	return w.Watcher_moby27782.Events
}

func watchFile_moby27782() FileWatcher_moby27782 {
	fileWatcher := New_moby27782()
	return fileWatcher
}

type LogWatcher_moby27782 struct {
	closeOnce     sync.Once
	closeNotifier chan struct{}
}

func (w *LogWatcher_moby27782) Close() {
	w.closeOnce.Do(func() {
		close(w.closeNotifier)
	})
}

func (w *LogWatcher_moby27782) WatchClose() <-chan struct{} {
	return w.closeNotifier
}

func NewLogWatcher_moby27782() *LogWatcher_moby27782 {
	return &LogWatcher_moby27782{
		closeNotifier: make(chan struct{}),
	}
}

func followLogs_moby27782(logWatcher *LogWatcher_moby27782) {
	fileWatcher := watchFile_moby27782()
	defer func() {
		fileWatcher.Close()
	}()
	waitRead := func() {
		runtime.Gosched()
		select {
		case <-fileWatcher.Events():
		case <-logWatcher.WatchClose():
			fileWatcher.Remove()
			return
		}
	}
	handleDecodeErr := func() {
		waitRead()
	}
	handleDecodeErr()
}

type Container_moby27782 struct {
	LogDriver *JSONFileLogger_moby27782
}

func (container *Container_moby27782) InitializeStdio() {
	if err := container.startLogging(); err != nil {
		container.Reset()
	}
}

func (container *Container_moby27782) startLogging() error {
	l := &JSONFileLogger_moby27782{
		readers: make(map[*LogWatcher_moby27782]struct{}),
	}
	container.LogDriver = l
	l.ReadLogs()
	return errors.New("Some error")
}

func (container *Container_moby27782) Reset() {
	if container.LogDriver != nil {
		container.LogDriver.Close()
	}
}

type JSONFileLogger_moby27782 struct {
	mu      sync.Mutex
	readers map[*LogWatcher_moby27782]struct{}
}

func (l *JSONFileLogger_moby27782) ReadLogs() *LogWatcher_moby27782 {
	logWatcher := NewLogWatcher_moby27782()
	go l.readLogs(logWatcher) // G2
	return logWatcher
}

func (l *JSONFileLogger_moby27782) readLogs(logWatcher *LogWatcher_moby27782) {
	l.mu.Lock()
	defer l.mu.Unlock()

	l.readers[logWatcher] = struct{}{}
	followLogs_moby27782(logWatcher)
}

func (l *JSONFileLogger_moby27782) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()

	for r := range l.readers {
		r.Close()
		delete(l.readers, r)
	}
}

func Moby27782() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 10000; i++ {
		go (&Container_moby27782{}).InitializeStdio() // G1
	}
}
