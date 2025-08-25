/*
 * Project: kubernetes
 * Tag: Reproduce misbehavior
 * Issue or PR  : https://github.com/kubernetes/kubernetes/pull/58107
 * Buggy version: 2f17d782eb2772d6401da7ddced9ac90656a7a79
 * fix commit-id: 010a127314a935d8d038f8dd4559fc5b249813e4
 * Flaky: 53/100
 * Description:
 *   The rules for read and write lock: allows concurrent read lock;
 * write lock has higher priority than read lock.
 *   There are two queues (queue 1 and queue 2) involved in this bug,
 * and the two queues are protected by the same read-write lock
 * (rq.workerLock.RLock()). Before getting an element from queue 1 or
 * queue 2, rq.workerLock.RLock() is acquired. If the queue is empty,
 * cond.Wait() will be invoked. There is another goroutine (goroutine D),
 * which will periodically invoke rq.workerLock.Lock(). Under the following
 * situation, deadlock will happen. Queue 1 is empty, so that some goroutines
 * hold rq.workerLock.RLock(), and block at cond.Wait(). Goroutine D is
 * blocked when acquiring rq.workerLock.Lock(). Some goroutines try to process
 * jobs in queue 2, but they are blocked when acquiring rq.workerLock.RLock(),
 * since write lock has a higher priority.
 *   The fix is to not acquire rq.workerLock.RLock(), while pulling data
 * from any queue. Therefore, when a goroutine is blocked at cond.Wait(),
 * rq.workLock.RLock() is not held.
 */

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"sync"
	"time"
)

func init() {
	register("Kubernetes58107", Kubernetes58107)
}

type RateLimitingInterface_kubernetes58107 interface {
	Get()
	Put()
}

type Type_kubernetes58107 struct {
	cond *sync.Cond
}

func (q *Type_kubernetes58107) Get() {
	q.cond.L.Lock()
	defer q.cond.L.Unlock()
	q.cond.Wait()
}

func (q *Type_kubernetes58107) Put() {
	q.cond.Signal()
}

type ResourceQuotaController_kubernetes58107 struct {
	workerLock        sync.RWMutex
	queue             RateLimitingInterface_kubernetes58107
	missingUsageQueue RateLimitingInterface_kubernetes58107
}

func (rq *ResourceQuotaController_kubernetes58107) worker(queue RateLimitingInterface_kubernetes58107, _ string) {
	workFunc := func() bool {
		rq.workerLock.RLock()
		defer rq.workerLock.RUnlock()
		queue.Get()
		return true
	}
	for {
		if quit := workFunc(); quit {
			return
		}
	}
}

func (rq *ResourceQuotaController_kubernetes58107) Run() {
	// deadlocks: x > 0
	go rq.worker(rq.queue, "G1") // G3
	// deadlocks: x > 0
	go rq.worker(rq.missingUsageQueue, "G2") // G4
}

func (rq *ResourceQuotaController_kubernetes58107) Sync() {
	for i := 0; i < 100000; i++ {
		rq.workerLock.Lock()
		runtime.Gosched()
		rq.workerLock.Unlock()
	}
}

func (rq *ResourceQuotaController_kubernetes58107) HelperSignals() {
	for i := 0; i < 100000; i++ {
		rq.queue.Put()
		rq.missingUsageQueue.Put()
	}
}

func startResourceQuotaController_kubernetes58107() {
	resourceQuotaController := &ResourceQuotaController_kubernetes58107{
		queue:             &Type_kubernetes58107{sync.NewCond(&sync.Mutex{})},
		missingUsageQueue: &Type_kubernetes58107{sync.NewCond(&sync.Mutex{})},
	}

	go resourceQuotaController.Run() // G2
	// deadlocks: x > 0
	go resourceQuotaController.Sync() // G5
	resourceQuotaController.HelperSignals()
}

func Kubernetes58107() {
	prof := pprof.Lookup("goroutineleak")
	defer func() {
		time.Sleep(1000 * time.Millisecond)
		prof.WriteTo(os.Stdout, 2)
	}()

	for i := 0; i < 1000; i++ {
		go startResourceQuotaController_kubernetes58107() // G1
	}
}

// Example of deadlock:
//
// 	G1 								G3							G4							G5
//	------------------------------------------------------------------------------------------------------------
// 	<<<done>>> (no more signals)	...							...							Sync()
// 									rq.workerLock.RLock()		.							.
// 									q.cond.L.Lock()				.							.
// 									q.cond.Wait()				.							.
// 									.							.							rq.workerLock.Lock()
// 									.							rq.workerLock.RLock()		.
// 									.							q.cond.L.Lock()				.
//	--------------------------------------------G3, G4, G5 leak-------------------------------------------------
