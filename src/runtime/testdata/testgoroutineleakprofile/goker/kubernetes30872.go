// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a MIT
// license that can be found in the LICENSE file.

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes30872", Kubernetes30872)
}

type PopProcessFunc_kubernetes30872 func()

type ProcessFunc_kubernetes30872 func()

func Util_kubernetes30872(f func(), stopCh <-chan struct{}) {
	JitterUntil_kubernetes30872(f, stopCh)
}

func JitterUntil_kubernetes30872(f func(), stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			return
		default:
		}
		func() {
			f()
		}()
	}
}

type Queue_kubernetes30872 interface {
	HasSynced()
	Pop(PopProcessFunc_kubernetes30872)
}

type Config_kubernetes30872 struct {
	Queue   Queue_kubernetes30872
	Process ProcessFunc_kubernetes30872
}

type Controller_kubernetes30872 struct {
	config Config_kubernetes30872
}

func (c *Controller_kubernetes30872) Run(stopCh <-chan struct{}) {
	Util_kubernetes30872(c.processLoop, stopCh)
}

func (c *Controller_kubernetes30872) HasSynced() {
	c.config.Queue.HasSynced()
}

func (c *Controller_kubernetes30872) processLoop() {
	c.config.Queue.Pop(PopProcessFunc_kubernetes30872(c.config.Process))
}

type ControllerInterface_kubernetes30872 interface {
	Run(<-chan struct{})
	HasSynced()
}

type ResourceEventHandler_kubernetes30872 interface {
	OnAdd()
}

type ResourceEventHandlerFuncs_kubernetes30872 struct {
	AddFunc func()
}

func (r ResourceEventHandlerFuncs_kubernetes30872) OnAdd() {
	if r.AddFunc != nil {
		r.AddFunc()
	}
}

type informer_kubernetes30872 struct {
	controller ControllerInterface_kubernetes30872

	stopChan chan struct{}
}

type federatedInformerImpl_kubernetes30872 struct {
	sync.Mutex
	clusterInformer informer_kubernetes30872
}

func (f *federatedInformerImpl_kubernetes30872) ClustersSynced() {
	f.Lock() // L1
	defer f.Unlock()
	f.clusterInformer.controller.HasSynced()
}

func (f *federatedInformerImpl_kubernetes30872) addCluster() {
	f.Lock() // L1
	defer f.Unlock()
}

func (f *federatedInformerImpl_kubernetes30872) Start() {
	f.Lock() // L1
	defer f.Unlock()

	f.clusterInformer.stopChan = make(chan struct{})
	go f.clusterInformer.controller.Run(f.clusterInformer.stopChan) // G2
	runtime.Gosched()
}

func (f *federatedInformerImpl_kubernetes30872) Stop() {
	f.Lock() // L1
	defer f.Unlock()
	close(f.clusterInformer.stopChan)
}

type DelayingDeliverer_kubernetes30872 struct{}

func (d *DelayingDeliverer_kubernetes30872) StartWithHandler(handler func()) {
	go func() { // G4
		handler()
	}()
}

type FederationView_kubernetes30872 interface {
	ClustersSynced()
}

type FederatedInformer_kubernetes30872 interface {
	FederationView_kubernetes30872
	Start()
	Stop()
}

type NamespaceController_kubernetes30872 struct {
	namespaceDeliverer         *DelayingDeliverer_kubernetes30872
	namespaceFederatedInformer FederatedInformer_kubernetes30872
}

func (nc *NamespaceController_kubernetes30872) isSynced() {
	nc.namespaceFederatedInformer.ClustersSynced()
}

func (nc *NamespaceController_kubernetes30872) reconcileNamespace() {
	nc.isSynced()
}

func (nc *NamespaceController_kubernetes30872) Run(stopChan <-chan struct{}) {
	nc.namespaceFederatedInformer.Start()
	go func() { // G3
		<-stopChan
		nc.namespaceFederatedInformer.Stop()
	}()
	nc.namespaceDeliverer.StartWithHandler(func() {
		nc.reconcileNamespace()
	})
}

type DeltaFIFO_kubernetes30872 struct {
	lock sync.RWMutex
}

func (f *DeltaFIFO_kubernetes30872) HasSynced() {
	f.lock.Lock() // L2
	defer f.lock.Unlock()
}

func (f *DeltaFIFO_kubernetes30872) Pop(process PopProcessFunc_kubernetes30872) {
	f.lock.Lock() // L2
	defer f.lock.Unlock()
	process()
}

func NewFederatedInformer_kubernetes30872() FederatedInformer_kubernetes30872 {
	federatedInformer := &federatedInformerImpl_kubernetes30872{}
	federatedInformer.clusterInformer.controller = NewInformer_kubernetes30872(
		ResourceEventHandlerFuncs_kubernetes30872{
			AddFunc: func() {
				federatedInformer.addCluster()
			},
		})
	return federatedInformer
}

func NewInformer_kubernetes30872(h ResourceEventHandler_kubernetes30872) *Controller_kubernetes30872 {
	fifo := &DeltaFIFO_kubernetes30872{}
	cfg := &Config_kubernetes30872{
		Queue: fifo,
		Process: func() {
			h.OnAdd()
		},
	}
	return &Controller_kubernetes30872{config: *cfg}
}

func NewNamespaceController_kubernetes30872() *NamespaceController_kubernetes30872 {
	nc := &NamespaceController_kubernetes30872{}
	nc.namespaceDeliverer = &DelayingDeliverer_kubernetes30872{}
	nc.namespaceFederatedInformer = NewFederatedInformer_kubernetes30872()
	return nc
}

func Kubernetes30872() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 100; i++ {
		go func() { // G1
			namespaceController := NewNamespaceController_kubernetes30872()
			stop := make(chan struct{})
			namespaceController.Run(stop)
			close(stop)
		}()
	}
}
