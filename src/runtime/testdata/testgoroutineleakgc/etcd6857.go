/*
 * Project: etcd
 * Issue or PR  : https://github.com/etcd-io/etcd/pull/6857
 * Buggy version: 7c8f13aed7fe251e7066ed6fc1a090699c2cae0e
 * fix commit-id: 7afc490c95789c408fbc256d8e790273d331c984
 * Flaky: 19/100
 */
package main

import (
	"runtime"
	"time"
)

func init() {
	register("Etcd6857", Etcd6857)
}

type Status_etcd6857 struct{}

type node_etcd6857 struct {
	status chan chan Status_etcd6857
	stop   chan struct{}
	done   chan struct{}
}

func (n *node_etcd6857) Status() Status_etcd6857 {
	c := make(chan Status_etcd6857)
	n.status <- c
	return <-c
}

func (n *node_etcd6857) run() {
	for {
		select {
		case c := <-n.status:
			c <- Status_etcd6857{}
		case <-n.stop:
			close(n.done)
			return
		}
	}
}

func (n *node_etcd6857) Stop() {
	select {
	case n.stop <- struct{}{}:
	case <-n.done:
		return
	}
	<-n.done
}

func NewNode_etcd6857() *node_etcd6857 {
	return &node_etcd6857{
		status: make(chan chan Status_etcd6857),
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

///
/// G1				G2				G3
/// n.run()
///									n.Stop()
///									n.stop<-
/// <-n.stop
///									<-n.done
/// close(n.done)
///	return
///									return
///					n.Status()
///					n.status<-
///----------------G2 leak-------------------
///

func Etcd6857() {
	defer func() {
		time.Sleep(100 * time.Millisecond)
		runtime.GC()
	}()
	for i := 0; i <= 100; i++ {
		go func() {
			n := NewNode_etcd6857()
			go n.run() // G1
			// deadlocks: x > 0
			go n.Status() // G2
			go n.Stop()   // G3
		}()
	}
}
