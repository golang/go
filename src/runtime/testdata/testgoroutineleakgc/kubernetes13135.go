/*
 * Project: kubernetes
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/13135
 * Buggy version: 6ced66249d4fd2a81e86b4a71d8df0139fe5ceae
 * fix commit-id: a12b7edc42c5c06a2e7d9f381975658692951d5a
 * Flaky: 93/100
 */
package main

import (
	"os"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes13135", Kubernetes13135)
}

var (
	StopChannel_kubernetes13135 chan struct{}
)

func Util_kubernetes13135(f func(), period time.Duration, stopCh <-chan struct{}) {
	for {
		select {
		case <-stopCh:
			return
		default:
		}
		func() {
			f()
		}()
		time.Sleep(period)
	}
}

type Store_kubernetes13135 interface {
	Add(obj interface{})
	Replace(obj interface{})
}

type Reflector_kubernetes13135 struct {
	store Store_kubernetes13135
}

func (r *Reflector_kubernetes13135) ListAndWatch(stopCh <-chan struct{}) error {
	r.syncWith()
	return nil
}

func NewReflector_kubernetes13135(store Store_kubernetes13135) *Reflector_kubernetes13135 {
	return &Reflector_kubernetes13135{
		store: store,
	}
}

func (r *Reflector_kubernetes13135) syncWith() {
	r.store.Replace(nil)
}

type Cacher_kubernetes13135 struct {
	sync.Mutex
	initialized sync.WaitGroup
	initOnce    sync.Once
	watchCache  *WatchCache_kubernetes13135
	reflector   *Reflector_kubernetes13135
}

func (c *Cacher_kubernetes13135) processEvent() {
	c.Lock()
	defer c.Unlock()
}

func (c *Cacher_kubernetes13135) startCaching(stopChannel <-chan struct{}) {
	c.Lock()
	for {
		err := c.reflector.ListAndWatch(stopChannel)
		if err == nil {
			break
		}
	}
}

type WatchCache_kubernetes13135 struct {
	sync.RWMutex
	onReplace func()
	onEvent   func()
}

func (w *WatchCache_kubernetes13135) SetOnEvent(onEvent func()) {
	w.Lock()
	defer w.Unlock()
	w.onEvent = onEvent
}

func (w *WatchCache_kubernetes13135) SetOnReplace(onReplace func()) {
	w.Lock()
	defer w.Unlock()
	w.onReplace = onReplace
}

func (w *WatchCache_kubernetes13135) processEvent() {
	w.Lock()
	defer w.Unlock()
	if w.onEvent != nil {
		w.onEvent()
	}
}

func (w *WatchCache_kubernetes13135) Add(obj interface{}) {
	w.processEvent()
}

func (w *WatchCache_kubernetes13135) Replace(obj interface{}) {
	w.Lock()
	defer w.Unlock()
	if w.onReplace != nil {
		w.onReplace()
	}
}

func NewCacher_kubernetes13135() *Cacher_kubernetes13135 {
	watchCache := &WatchCache_kubernetes13135{}
	cacher := &Cacher_kubernetes13135{
		initialized: sync.WaitGroup{},
		watchCache:  watchCache,
		reflector:   NewReflector_kubernetes13135(watchCache),
	}
	cacher.initialized.Add(1)
	watchCache.SetOnReplace(func() {
		cacher.initOnce.Do(func() { cacher.initialized.Done() })
		cacher.Unlock()
	})
	watchCache.SetOnEvent(cacher.processEvent)
	stopCh := StopChannel_kubernetes13135
	go Util_kubernetes13135(func() { cacher.startCaching(stopCh) }, 0, stopCh) // G2
	cacher.initialized.Wait()
	return cacher
}

///
/// G1								G2								G3
/// NewCacher()
/// watchCache.SetOnReplace()
/// watchCache.SetOnEvent()
/// 								cacher.startCaching()
///									c.Lock()
/// 								c.reflector.ListAndWatch()
/// 								r.syncWith()
/// 								r.store.Replace()
/// 								w.Lock()
/// 								w.onReplace()
/// 								cacher.initOnce.Do()
/// 								cacher.Unlock()
/// return cacher
///																	c.watchCache.Add()
///																	w.processEvent()
///																	w.Lock()
///									cacher.startCaching()
///									c.Lock()
///									...
///																	c.Lock()
///									w.Lock()
///--------------------------------G2,G3 deadlock-------------------------------------
///

func Kubernetes13135() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(100 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	StopChannel_kubernetes13135 = make(chan struct{})
	for i := 0; i < 50; i++ {
		go func() {
			// deadlocks: x > 0
			c := NewCacher_kubernetes13135() // G1
			go c.watchCache.Add(nil)         // G3
		}()
	}
	go close(StopChannel_kubernetes13135)
}
