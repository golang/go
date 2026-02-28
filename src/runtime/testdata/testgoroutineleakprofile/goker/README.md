# GoKer

The following examples are obtained from the publication
"GoBench: A Benchmark Suite of Real-World Go Concurrency Bugs"
(doi:10.1109/CGO51591.2021.9370317).

**Authors**
Ting Yuan (yuanting@ict.ac.cn):
    State Key Laboratory of Computer Architecture, Institute of Computing Technology, Chinese Academy of Sciences,
    University of Chinese Academy of Sciences, Beijing, China;
Guangwei Li (liguangwei@ict.ac.cn):
    State Key Laboratory of Computer Architecture, Institute of Computing Technology, Chinese Academy of Sciences,
    University of Chinese Academy of Sciences, Beijing, China;
Jie Lu† (lujie@ict.ac.an):
    State Key Laboratory of Computer Architecture, Institute of Computing Technology, Chinese Academy of Sciences;
Chen Liu (liuchen17z@ict.ac.cn):
    State Key Laboratory of Computer Architecture, Institute of Computing Technology, Chinese Academy of Sciences,
    University of Chinese Academy of Sciences, Beijing, China
Lian Li (lianli@ict.ac.cn):
    State Key Laboratory of Computer Architecture, Institute of Computing Technology, Chinese Academy of Sciences,
    University of Chinese Academy of Sciences, Beijing, China;
Jingling Xue (jingling@cse.unsw.edu.au):
    University of New South Wales, School of Computer Science and Engineering, Sydney, Australia

White paper: https://lujie.ac.cn/files/papers/GoBench.pdf

The examples have been modified in order to run the goroutine leak profiler.
Buggy snippets are moved from within a unit test to separate applications.
Each is then independently executed, possibly as multiple copies within the
same application in order to exercise more interleavings. Concurrently, the
main program sets up a waiting period (typically 1ms), followed by a goroutine
leak profile request. Other modifications may involve injecting calls to
`runtime.Gosched()`, to more reliably exercise buggy interleavings, or reductions
in waiting periods when calling `time.Sleep`, in order to reduce overall testing
time.

The resulting goroutine leak profile is analyzed to ensure that no unexpecte
leaks occurred, and that the expected leaks did occur. If the leak is flaky, the
only purpose of the expected leak list is to protect against unexpected leaks.

The examples have also been modified to remove data races, since those create flaky
test failures, when really all we care about are leaked goroutines.

The entries below document each of the corresponding leaks.

## Cockroach/10214

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#10214]|[pull request]|[patch]| Resource | AB-BA leak |

[cockroach#10214]:(cockroach10214_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/10214/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/10214

### Description

This goroutine leak is caused by different order when acquiring
coalescedMu.Lock() and raftMu.Lock(). The fix is to refactor sendQueuedHeartbeats()
so that cockroachdb can unlock coalescedMu before locking raftMu.

### Example execution

```go
G1                                      G2
------------------------------------------------------------------------------------
s.sendQueuedHeartbeats()                .
s.coalescedMu.Lock() [L1]               .
s.sendQueuedHeartbeatsToNode()          .
s.mu.replicas[0].reportUnreachable()    .
s.mu.replicas[0].raftMu.Lock() [L2]     .
.                                       s.mu.replicas[0].tick()
.                                       s.mu.replicas[0].raftMu.Lock() [L2]
.                                       s.mu.replicas[0].tickRaftMuLocked()
.                                       s.mu.replicas[0].mu.Lock() [L3]
.                                       s.mu.replicas[0].maybeQuiesceLocked()
.                                       s.mu.replicas[0].maybeCoalesceHeartbeat()
.                                       s.coalescedMu.Lock() [L1]
--------------------------------G1,G2 leak------------------------------------------
```

## Cockroach/1055

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#1055]|[pull request]|[patch]| Mixed | Channel & WaitGroup |

[cockroach#1055]:(cockroach1055_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/1055/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/1055

### Description

1. `Stop()` is called and blocked at `s.stop.Wait()` after acquiring the lock.
2. `StartTask()` is called and attempts to acquire the lock. It is then blocked.
3. `Stop()` never finishes since the task doesn't call SetStopped.

### Example execution

```go
G1                      G2.0                       G2.1                       G2.2                       G3
-------------------------------------------------------------------------------------------------------------------------------
s[0].stop.Add(1) [1]
go func() [G2.0]
s[1].stop.Add(1) [1]    .
go func() [G2.1]        .
s[2].stop.Add(1) [1]    .                          .
go func() [G2.2]        .                          .
go func() [G3]          .                          .                          .
<-done                  .                          .                          .                          .
.                       s[0].StartTask()           .                          .                          .
.                       s[0].draining == 0         .                          .                          .
.                       .                          s[1].StartTask()           .                          .
.                       .                          s[1].draining == 0         .                          .
.                       .                          .                          s[2].StartTask()           .
.                       .                          .                          s[2].draining == 0         .
.                       .                          .                          .                          s[0].Quiesce()
.                       .                          .                          .                          s[0].mu.Lock() [L1[0]]
.                       s[0].mu.Lock() [L1[0]]     .                          .                          .
.                       s[0].drain.Add(1) [1]      .                          .                          .
.                       s[0].mu.Unlock() [L1[0]]   .                          .                          .
.                       <-s[0].ShouldStop()        .                          .                          .
.                       .                          .                          .                          s[0].draining = 1
.                       .                          .                          .                          s[0].drain.Wait()
.                       .                          s[0].mu.Lock() [L1[1]]     .                          .
.                       .                          s[1].drain.Add(1) [1]      .                          .
.                       .                          s[1].mu.Unlock() [L1[1]]   .                          .
.                       .                          <-s[1].ShouldStop()        .                          .
.                       .                          .                          s[2].mu.Lock() [L1[2]]     .
.                       .                          .                          s[2].drain.Add() [1]       .
.                       .                          .                          s[2].mu.Unlock() [L1[2]]   .
.                       .                          .                          <-s[2].ShouldStop()        .
----------------------------------------------------G1, G2.[0..2], G3 leak-----------------------------------------------------
```

## Cockroach/10790

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#10790]|[pull request]|[patch]| Communication | Channel & Context |

[cockroach#10790]:(cockroach10790_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/10790/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/10790

### Description

It is possible that a message from `ctxDone` will make `beginCmds`
return without draining the channel `ch`, so that anonymous function
goroutines will leak.

### Example execution

```go
G1                  G2              helper goroutine
-----------------------------------------------------
.                   .               r.sendChans()
r.beginCmds()       .               .
.                   .               ch1 <- true
<- ch1              .               .
.                   .               ch2 <- true
...
.                   cancel()
<- ch1
------------------G1 leak----------------------------
```

## Cockroach/13197

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#13197]|[pull request]|[patch]| Communication | Channel & Context |

[cockroach#13197]:(cockroach13197_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/13197/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/13197

### Description

One goroutine executing `(*Tx).awaitDone()` blocks and
waiting for a signal `context.Done()`.

### Example execution

```go
G1              G2
-------------------------------
begin()
.               awaitDone()
return          .
.               <-tx.ctx.Done()
-----------G2 leaks------------
```

## Cockroach/13755

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#13755]|[pull request]|[patch]| Communication | Channel & Context |

[cockroach#13755]:(cockroach13755_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/13755/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/13755

### Description

The buggy code does not close the db query result (rows),
so that one goroutine running `(*Rows).awaitDone` is blocked forever.
The blocking goroutine is waiting for cancel signal from context.

### Example execution

```go
G1                      G2
---------------------------------------
initContextClose()
.                       awaitDone()
return                  .
.                       <-tx.ctx.Done()
---------------G2 leaks----------------
```

## Cockroach/1462

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#1462]|[pull request]|[patch]| Mixed | Channel & WaitGroup |

[cockroach#1462]:(cockroach1462_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/1462/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/1462

### Description

Executing `<-stopper.ShouldStop()` in `processEventsUntil` may cause
goroutines created by `lt.RunWorker` in `lt.start` to be stuck sending
a message over `lt.Events`. The main thread is then stuck at `s.stop.Wait()`,
since the sender goroutines cannot call `s.stop.Done()`.

### Example execution

```go
G1                                  G2                                G3
-------------------------------------------------------------------------------------------------------
NewLocalInterceptableTransport()
lt.start()
lt.stopper.RunWorker()
s.AddWorker()
s.stop.Add(1) [1]
go func() [G2]
stopper.RunWorker()                 .
s.AddWorker()                       .
s.stop.Add(1) [2]                   .
go func() [G3]                      .
s.Stop()                            .                                 .
s.Quiesce()                         .                                 .
.                                   select [default]                  .
.                                   lt.Events <- interceptMessage(0)  .
close(s.stopper)                    .                                 .
.                                   .                                 select [<-stopper.ShouldStop()]
.                                   .                                 <<<done>>>
s.stop.Wait()                       .
----------------------------------------------G1,G2 leak-----------------------------------------------
```

## Cockroach/16167

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#16167]|[pull request]|[patch]| Resource | Double Locking |

[cockroach#16167]:(cockroach16167_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/16167/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/16167

### Description

This is another example of goroutine leaks caused by recursively
acquiring `RWLock`.
There are two lock variables (`systemConfigCond` and `systemConfigMu`)
which refer to the same underlying lock. The leak invovlves two goroutines.
The first acquires `systemConfigMu.Lock()`, then tries to acquire `systemConfigMu.RLock()`.
The second acquires `systemConfigMu.Lock()`.
If the second goroutine interleaves in between the two lock operations of the
first goroutine, both goroutines will leak.

### Example execution

```go
G1                                G2
---------------------------------------------------------------
.                                 e.Start()
.                                 e.updateSystemConfig()
e.execParsed()                    .
e.systemConfigCond.L.Lock() [L1]  .
.                                 e.systemConfigMu.Lock() [L1]
e.systemConfigMu.RLock() [L1]     .
------------------------G1,G2 leak-----------------------------
```

## Cockroach/18101

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#18101]|[pull request]|[patch]| Resource | Double Locking |

[cockroach#18101]:(cockroach18101_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/18101/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/18101

### Description

The `context.Done()` signal short-circuits the reader goroutine, but not
the senders, leading them to leak.

### Example execution

```go
G1                      G2                   helper goroutine
--------------------------------------------------------------
restore()
.                       splitAndScatter()
<-readyForImportCh      .
<-readyForImportCh <==> readyForImportCh<-
...
.                       .                    cancel()
<<done>>                .                    <<done>>
                        readyForImportCh<-
-----------------------G2 leaks--------------------------------
```

## Cockroach/2448

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#2448]|[pull request]|[patch]| Communication | Channel |

[cockroach#2448]:(cockroach2448_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/2448/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/2448

### Description

This bug is caused by two goroutines waiting for each other
to unblock their channels:

1) `MultiRaft` sends the commit event for the Membership change
2) `store.processRaft` takes it and begins processing
3) another command commits and triggers another `sendEvent`, but
   this blocks since `store.processRaft` isn't ready for another
   `select`. Consequently the main `MultiRaft` loop is waiting for
   that as well.
4) the `Membership` change was applied to the range, and the store
   now tries to execute the callback
5) the callback tries to write to `callbackChan`, but that is
   consumed by the `MultiRaft` loop, which is currently waiting
   for `store.processRaft` to consume from the events channel,
   which it will only do after the callback has completed.

### Example execution

```go
G1                          G2
--------------------------------------------------------------------------
s.processRaft()             st.start()
select                      .
.                           select [default]
.                           s.handleWriteResponse()
.                           s.sendEvent()
.                           select
<-s.multiraft.Events <----> m.Events <- event
.                           select [default]
.                           s.handleWriteResponse()
.                           s.sendEvent()
.                           select [m.Events<-, <-s.stopper.ShouldStop()]
callback()                  .
select [
   m.callbackChan<-,
   <-s.stopper.ShouldStop()
]                           .
------------------------------G1,G2 leak----------------------------------
```

## Cockroach/24808

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#24808]|[pull request]|[patch]| Communication | Channel |

[cockroach#24808]:(cockroach24808_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/24808/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/24808

### Description

When we `Start` the `Compactor`, it may already have received
`Suggestions`, leaking the previously blocking write to a full channel.

### Example execution

```go
G1
------------------------------------------------
...
compactor.ch <-
compactor.Start()
compactor.ch <-
--------------------G1 leaks--------------------
```

## Cockroach/25456

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#25456]|[pull request]|[patch]| Communication | Channel |

[cockroach#25456]:(cockroach25456_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/25456/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/25456

### Description

When `CheckConsistency` (in the complete code) returns an error, the queue
checks whether the store is draining to decide whether the error is worth
logging. This check was incorrect and would block until the store actually
started draining.

### Example execution

```go
G1
---------------------------------------
...
<-repl.store.Stopper().ShouldQuiesce()
---------------G1 leaks----------------
```

## Cockroach/35073

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#35073]|[pull request]|[patch]| Communication | Channel |

[cockroach#35073]:(cockroach35073_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/35073/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/35073

### Description

Previously, the outbox could fail during startup without closing its
`RowChannel`. This could lead to goroutine leaks in rare cases due
to channel communication mismatch.

### Example execution

## Cockroach/35931

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#35931]|[pull request]|[patch]| Communication | Channel |

[cockroach#35931]:(cockroach35931_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/35931/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/35931

### Description

Previously, if a processor that reads from multiple inputs was waiting
on one input to provide more data, and the other input was full, and
both inputs were connected to inbound streams, it was possible to
cause goroutine leaks during flow cancellation when trying to propagate
the cancellation metadata messages into the flow. The cancellation method
wrote metadata messages to each inbound stream one at a time, so if the
first one was full, the canceller would block and never send a cancellation
message to the second stream, which was the one actually being read from.

### Example execution

## Cockroach/3710

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#3710]|[pull request]|[patch]| Resource | RWR Deadlock |

[cockroach#3710]:(cockroach3710_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/3710/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/3710

### Description

The goroutine leak is caused by acquiring a RLock twice in a call chain.
`ForceRaftLogScanAndProcess(acquire s.mu.RLock())`
`-> MaybeAdd()`
`-> shouldQueue()`
`-> getTruncatableIndexes()`
`->RaftStatus(acquire s.mu.Rlock())`

### Example execution

```go
G1                                     	G2
------------------------------------------------------------
store.ForceRaftLogScanAndProcess()
s.mu.RLock()
s.raftLogQueue.MaybeAdd()
bq.impl.shouldQueue()
getTruncatableIndexes()
r.store.RaftStatus()
.                                        store.processRaft()
.                                        s.mu.Lock()
s.mu.RLock()
----------------------G1,G2 leak-----------------------------
```

## Cockroach/584

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#584]|[pull request]|[patch]| Resource | Double Locking |

[cockroach#584]:(cockroach584_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/584/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/584

### Description

Missing call to `mu.Unlock()` before the `break` in the loop.

### Example execution

```go
G1
---------------------------
g.bootstrap()
g.mu.Lock() [L1]
if g.closed { ==> break
g.manage()
g.mu.Lock() [L1]
----------G1 leaks---------
```

## Cockroach/6181

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#6181]|[pull request]|[patch]| Resource | RWR Deadlock |

[cockroach#6181]:(cockroach6181_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/6181/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/6181

### Description

The same `RWMutex` may be recursively acquired for both reading and writing.

### Example execution

```go
G1                                G2                               G3                     ...
-----------------------------------------------------------------------------------------------
testRangeCacheCoalescedRquests()
initTestDescriptorDB()
pauseLookupResumeAndAssert()
return
.                                 doLookupWithToken()
.                                 .                                doLookupWithToken()
.                                 rc.LookupRangeDescriptor()       .
.                                 .                                rc.LookupRangeDescriptor()
.                                 rdc.rangeCacheMu.RLock()         .
.                                 rdc.String()                     .
.                                 .                                rdc.rangeCacheMu.RLock()
.                                 .                                fmt.Printf()
.                                 .                                rdc.rangeCacheMu.RUnlock()
.                                 .                                rdc.rangeCacheMu.Lock()
.                                 rdc.rangeCacheMu.RLock()         .
-----------------------------------G2,G3,... leak----------------------------------------------
```

## Cockroach/7504

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#7504]|[pull request]|[patch]| Resource | AB-BA Deadlock |

[cockroach#7504]:(cockroach7504_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/7504/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/7504

### Description

The locks are acquired as `leaseState` and `tableNameCache` in `Release()`, but
as `tableNameCache` and `leaseState` in `AcquireByName`, leading to an AB-BA deadlock.

### Example execution

```go
G1                        G2
-----------------------------------------------------
mgr.AcquireByName()       mgr.Release()
m.tableNames.get(id)      .
c.mu.Lock() [L2]          .
.                         t.release(lease)
.                         t.mu.Lock() [L3]
.                         s.mu.Lock() [L1]
lease.mu.Lock() [L1]      .
.                         t.removeLease(s)
.                         t.tableNameCache.remove()
.                         c.mu.Lock() [L2]
---------------------G1, G2 leak---------------------
```

## Cockroach/9935

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[cockroach#9935]|[pull request]|[patch]| Resource | Double Locking |

[cockroach#9935]:(cockroach9935_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/9935/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/9935

### Description

This bug is caused by acquiring `l.mu.Lock()` twice.

### Example execution

## Etcd/10492

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#10492]|[pull request]|[patch]| Resource | Double locking |

[etcd#10492]:(etcd10492_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/10492/files
[pull request]:https://github.com/etcd-io/etcd/pull/10492

### Description

A simple double locking case for lines 19, 31.

## Etcd/5509

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#5509]|[pull request]|[patch]| Resource | Double locking |

[etcd#5509]:(etcd5509_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/5509/files
[pull request]:https://github.com/etcd-io/etcd/pull/5509

### Description

`r.acquire()` returns holding `r.client.mu.RLock()` on a failure path (line 42).
This causes any call to `client.Close()` to leak goroutines.

## Etcd/6708

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#6708]|[pull request]|[patch]| Resource | Double locking |

[etcd#6708]:(etcd6708_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6708/files
[pull request]:https://github.com/etcd-io/etcd/pull/6708

### Description

Line 54, 49 double locking

## Etcd/6857

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#6857]|[pull request]|[patch]| Communication | Channel |

[etcd#6857]:(etcd6857_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6857/files
[pull request]:https://github.com/etcd-io/etcd/pull/6857

### Description

Choosing a different case in a `select` statement (`n.stop`) will
lead to goroutine leaks when sending over `n.status`.

### Example execution

```go
G1            	G2            	G3
-------------------------------------------
n.run()         .               .
.               .               n.Stop()
.               .               n.stop<-
<-n.stop        .               .
.               .               <-n.done
close(n.done)   .               .
return          .               .
.               .               return
.           	n.Status()
.           	n.status<-
----------------G2 leaks-------------------
```

## Etcd/6873

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#6873]|[pull request]|[patch]| Mixed | Channel & Lock |

[etcd#6873]:(etcd6873_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6873/files
[pull request]:https://github.com/etcd-io/etcd/pull/6873

### Description

This goroutine leak involves a goroutine acquiring a lock and being
blocked over a channel operation with no partner, while another tries
to acquire the same lock.

### Example execution

```go
G1                      G2                  G3
--------------------------------------------------------------
newWatchBroadcasts()
wbs.update()
wbs.updatec <-
return
.                       <-wbs.updatec       .
.                       wbs.coalesce()      .
.                       .                   wbs.stop()
.                       .                   wbs.mu.Lock()
.                       .                   close(wbs.updatec)
.                       .                   <-wbs.donec
.                       wbs.mu.Lock()       .
---------------------G2,G3 leak--------------------------------
```

## Etcd/7492

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#7492]|[pull request]|[patch]| Mixed | Channel & Lock |

[etcd#7492]:(etcd7492_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/7492/files
[pull request]:https://github.com/etcd-io/etcd/pull/7492

### Description

This goroutine leak involves a goroutine acquiring a lock and being
blocked over a channel operation with no partner, while another tries
to acquire the same lock.

### Example execution

```go
G2                                    G1
---------------------------------------------------------------
.                                     stk.run()
ts.assignSimpleTokenToUser()          .
t.simpleTokensMu.Lock()               .
t.simpleTokenKeeper.addSimpleToken()  .
tm.addSimpleTokenCh <- true           .
.                                     <-tm.addSimpleTokenCh
t.simpleTokensMu.Unlock()             .
ts.assignSimpleTokenToUser()          .
...
t.simpleTokensMu.Lock()
.                                     <-tokenTicker.C
tm.addSimpleTokenCh <- true           .
.                                     tm.deleteTokenFunc()
.                                     t.simpleTokensMu.Lock()
---------------------------G1,G2 leak--------------------------
```

## Etcd/7902

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[etcd#7902]|[pull request]|[patch]| Mixed | Channel & Lock |

[etcd#7902]:(etcd7902_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/7902/files
[pull request]:https://github.com/etcd-io/etcd/pull/7902

### Description

If the follower gooroutine acquires `mu.Lock()` first and calls
`rc.release()`, it will be blocked sending over `rcNextc`.
Only the leader can `close(nextc)` to unblock the follower.
However, in order to invoke `rc.release()`, the leader needs
to acquires `mu.Lock()`.
The fix is to remove the lock and unlock around `rc.release()`.

### Example execution

```go
G1                      G2 (leader)              G3 (follower)
---------------------------------------------------------------------
runElectionFunc()
doRounds()
wg.Wait()
.                       ...
.                       mu.Lock()
.                       rc.validate()
.                       rcNextc = nextc
.                       mu.Unlock()              ...
.                       .                        mu.Lock()
.                       .                        rc.validate()
.                       .                        mu.Unlock()
.                       .                        mu.Lock()
.                       .                        rc.release()
.                       .                        <-rcNextc
.                       mu.Lock()
-------------------------G1,G2,G3 leak--------------------------
```

## Grpc/1275

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#1275]|[pull request]|[patch]| Communication | Channel |

[grpc#1275]:(grpc1275_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1275/files
[pull request]:https://github.com/grpc/grpc-go/pull/1275

### Description

Two goroutines are involved in this leak. The main goroutine
is blocked at `case <- donec`, and is waiting for the second goroutine
to close the channel.
The second goroutine is created by the main goroutine. It is blocked
when calling `stream.Read()`, which invokes `recvBufferRead.Read()`.
The second goroutine is blocked at case `i := r.recv.get()`, and it is
waiting for someone to send a message to this channel.
It is the `client.CloseSream()` method called by the main goroutine that
should send the message, but it is not. The patch is to send out this message.

### Example execution

```go
G1                            G2
-----------------------------------------------------
testInflightStreamClosing()
.                             stream.Read()
.                             io.ReadFull()
.                             <-r.recv.get()
CloseStream()
<-donec
---------------------G1, G2 leak---------------------
```

## Grpc/1424

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#1424]|[pull request]|[patch]| Communication | Channel |

[grpc#1424]:(grpc1424_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1424/files
[pull request]:https://github.com/grpc/grpc-go/pull/1424

### Description

The goroutine running `cc.lbWatcher` returns without
draining the `done` channel.

### Example execution

```go
G1                      G2                          G3
-----------------------------------------------------------------
DialContext()           .                           .
.                       cc.dopts.balancer.Notify()  .
.                       .                           cc.lbWatcher()
.                       <-doneChan
close()
---------------------------G2 leaks-------------------------------
```

## Grpc/1460

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#1460]|[pull request]|[patch]| Mixed | Channel & Lock |

[grpc#1460]:(grpc1460_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1460/files
[pull request]:https://github.com/grpc/grpc-go/pull/1460

### Description

When gRPC keepalives are enabled (which isn't the case
by default at this time) and PermitWithoutStream is false
(the default), the client can leak goroutines when transitioning
between having no active stream and having one active
stream.The keepalive() goroutine is stuck at “<-t.awakenKeepalive”,
while the main goroutine is stuck in NewStream() on t.mu.Lock().

### Example execution

```go
G1                         G2
--------------------------------------------
client.keepalive()
.                       client.NewStream()
t.mu.Lock()
<-t.awakenKeepalive
.                       t.mu.Lock()
---------------G1,G2 leak-------------------
```

## Grpc/3017

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#3017]|[pull request]|[patch]| Resource | Missing unlock |

[grpc#3017]:(grpc3017_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/3017/files
[pull request]:https://github.com/grpc/grpc-go/pull/3017

### Description

Line 65 is an execution path with a missing unlock.

### Example execution

```go
G1                                  G2                                         G3
------------------------------------------------------------------------------------------------
NewSubConn([1])
ccc.mu.Lock() [L1]
sc = 1
ccc.subConnToAddr[1] = 1
go func() [G2]
<-done								.
.                                   ccc.RemoveSubConn(1)
.                                   ccc.mu.Lock()
.                                   addr = 1
.                                   entry = &subConnCacheEntry_grpc3017{}
.                                   cc.subConnCache[1] = entry
.                                   timer = time.AfterFunc() [G3]
.                                   entry.cancel = func()
.                                   sc = ccc.NewSubConn([1])
.                                   ccc.mu.Lock() [L1]
.                                   entry.cancel()
.                                   !timer.Stop() [true]
.                                   entry.abortDeleting = true
.                                   .                                          ccc.mu.Lock()
.                                   .                                          <<<done>>>
.                                   ccc.RemoveSubConn(1)
.                                   ccc.mu.Lock() [L1]
-------------------------------------------G1, G2 leak-----------------------------------------
```

## Grpc/660

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#660]|[pull request]|[patch]| Communication | Channel |

[grpc#660]:(grpc660_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/660/files
[pull request]:https://github.com/grpc/grpc-go/pull/660

### Description

The parent function could return without draining the done channel.

### Example execution

```go
G1                         G2               helper goroutine
-------------------------------------------------------------
doCloseLoopUnary()
.                                   	bc.stop <- true
<-bc.stop
return
.                       done <-
----------------------G2 leak--------------------------------
```

## Grpc/795

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#795]|[pull request]|[patch]| Resource | Double locking |

[grpc#795]:(grpc795_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/795/files
[pull request]:https://github.com/grpc/grpc-go/pull/795

### Description

Line 20 is an execution path with a missing unlock.

## Grpc/862

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[grpc#862]|[pull request]|[patch]| Communication | Channel & Context |

[grpc#862]:(grpc862_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/862/files
[pull request]:https://github.com/grpc/grpc-go/pull/862

### Description

When return value `conn` is `nil`, `cc(ClientConn)` is not closed.
The goroutine executing resetAddrConn is leaked. The patch is to
close `ClientConn` in `defer func()`.

### Example execution

```go
G1                  G2
---------------------------------------
DialContext()
.                   cc.resetAddrConn()
.                   resetTransport()
.                   <-ac.ctx.Done()
--------------G2 leak------------------
```

## Hugo/3251

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[hugo#3251]|[pull request]|[patch]| Resource | RWR deadlock |

[hugo#3251]:(hugo3251_test.go)
[patch]:https://github.com/gohugoio/hugo/pull/3251/files
[pull request]:https://github.com/gohugoio/hugo/pull/3251

### Description

A goroutine can hold `Lock()` at line 20 then acquire `RLock()` at
line 29. `RLock()` at line 29 will never be acquired because `Lock()`
at line 20 will never be released.

### Example execution

```go
G1                        G2                             G3
------------------------------------------------------------------------------------------
wg.Add(1) [W1: 1]
go func() [G2]
go func() [G3]
.                        resGetRemote()
.                        remoteURLLock.URLLock(url)
.                        l.Lock() [L1]
.                        l.m[url] = &sync.Mutex{} [L2]
.                        l.m[url].Lock() [L2]
.                        l.Unlock() [L1]
.                        .                        	     resGetRemote()
.                        .                        	     remoteURLLock.URLLock(url)
.                        .                        	     l.Lock() [L1]
.                        .                        	     l.m[url].Lock() [L2]
.                        remoteURLLock.URLUnlock(url)
.                        l.RLock() [L1]
...
wg.Wait() [W1]
----------------------------------------G1,G2,G3 leak--------------------------------------
```

## Hugo/5379

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[hugo#5379]|[pull request]|[patch]| Resource | Double locking |

[hugo#5379]:(hugo5379_test.go)
[patch]:https://github.com/gohugoio/hugo/pull/5379/files
[pull request]:https://github.com/gohugoio/hugo/pull/5379

### Description

A goroutine first acquire `contentInitMu` at line 99 then
acquire the same `Mutex` at line 66

## Istio/16224

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[istio#16224]|[pull request]|[patch]| Mixed | Channel & Lock |

[istio#16224]:(istio16224_test.go)
[patch]:https://github.com/istio/istio/pull/16224/files
[pull request]:https://github.com/istio/istio/pull/16224

### Description

A goroutine holds a `Mutex` at line 91 and is then blocked at line 93.
Another goroutine attempts to acquire the same `Mutex` at line 101 to
further drains the same channel at 103.

## Istio/17860

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[istio#17860]|[pull request]|[patch]| Communication | Channel |

[istio#17860]:(istio17860_test.go)
[patch]:https://github.com/istio/istio/pull/17860/files
[pull request]:https://github.com/istio/istio/pull/17860

### Description

`a.statusCh` can't be drained at line 70.

## Istio/18454

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[istio#18454]|[pull request]|[patch]| Communication | Channel & Context |

[istio#18454]:(istio18454_test.go)
[patch]:https://github.com/istio/istio/pull/18454/files
[pull request]:https://github.com/istio/istio/pull/18454

### Description

`s.timer.Stop()` at line 56 and 61 can be called concurrency
(i.e. from their entry point at line 104 and line 66).
See [Timer](https://golang.org/pkg/time/#Timer).

## Kubernetes/10182

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#10182]|[pull request]|[patch]| Mixed | Channel & Lock |

[kubernetes#10182]:(kubernetes10182_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/10182/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/10182

### Description

Goroutine 1 is blocked on a lock held by goroutine 3,
while goroutine 3 is blocked on sending message to `ch`,
which is read by goroutine 1.

### Example execution

```go
G1                         G2                        	G3
-------------------------------------------------------------------------------
s.Start()
s.syncBatch()
.                          s.SetPodStatus()
.                          s.podStatusesLock.Lock()
<-s.podStatusChannel <===> s.podStatusChannel <- true
.                          s.podStatusesLock.Unlock()
.                          return
s.DeletePodStatus()        .
.                          .                    	    s.podStatusesLock.Lock()
.                          .                    	    s.podStatusChannel <- true
s.podStatusesLock.Lock()
-----------------------------G1,G3 leak-----------------------------------------
```

## Kubernetes/11298

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#11298]|[pull request]|[patch]| Communication | Channel & Condition Variable |

[kubernetes#11298]:(kubernetes11298_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/11298/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/11298

### Description

`n.node` used the `n.lock` as underlaying locker. The service loop initially
locked it, the `Notify` function tried to lock it before calling `n.node.Signal()`,
leading to a goroutine leak. `n.cond.Signal()` at line 59 and line 81 are not
guaranteed to unblock the `n.cond.Wait` at line 56.

## Kubernetes/13135

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#13135]|[pull request]|[patch]| Resource | AB-BA deadlock |

[kubernetes#13135]:(kubernetes13135_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/13135/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/13135

### Description

```go
G1                              G2                              G3
----------------------------------------------------------------------------------
NewCacher()
watchCache.SetOnReplace()
watchCache.SetOnEvent()
.                               cacher.startCaching()
.                               c.Lock()
.                               c.reflector.ListAndWatch()
.                               r.syncWith()
.                               r.store.Replace()
.                               w.Lock()
.                               w.onReplace()
.                               cacher.initOnce.Do()
.                               cacher.Unlock()
return cacher                   .
.                               .                           	c.watchCache.Add()
.                               .                           	w.processEvent()
.                               .                           	w.Lock()
.                               cacher.startCaching()           .
.                               c.Lock()                        .
...
.                                                           	c.Lock()
.                               w.Lock()
--------------------------------G2,G3 leak-----------------------------------------
```

## Kubernetes/1321

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#1321]|[pull request]|[patch]| Mixed | Channel & Lock |

[kubernetes#1321]:(kubernetes1321_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/1321/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/1321

### Description

This is a lock-channel bug. The first goroutine invokes
`distribute()`, which holds `m.lock.Lock()`, while blocking
at sending message to `w.result`. The second goroutine
invokes `stopWatching()` function, which can unblock the first
goroutine by closing `w.result`. However, in order to close `w.result`,
`stopWatching()` function needs to acquire `m.lock.Lock()`.

The fix is to introduce another channel and put receive message
from the second channel in the same `select` statement as the
`w.result`. Close the second channel can unblock the first
goroutine, while no need to hold `m.lock.Lock()`.

### Example execution

```go
G1                          G2
----------------------------------------------
testMuxWatcherClose()
NewMux()
.                           m.loop()
.                           m.distribute()
.                           m.lock.Lock()
.                           w.result <- true
w := m.Watch()
w.Stop()
mw.m.stopWatching()
m.lock.Lock()
---------------G1,G2 leak---------------------
```

## Kubernetes/25331

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#25331]|[pull request]|[patch]| Communication | Channel & Context |

[kubernetes#25331]:(kubernetes25331_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/25331/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/25331

### Description

A potential goroutine leak occurs when an error has happened,
blocking `resultChan`, while cancelling context in `Stop()`.

### Example execution

```go
G1                    G2
------------------------------------
wc.run()
.                   wc.Stop()
.                   wc.errChan <-
.                   wc.cancel()
<-wc.errChan
wc.cancel()
wc.resultChan <-
-------------G1 leak----------------
```

## Kubernetes/26980

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#26980]|[pull request]|[patch]| Mixed | Channel & Lock |

[kubernetes#26980]:(kubernetes26980_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/26980/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/26980

### Description

A goroutine holds a `Mutex` at line 24 and blocked at line 35.
Another goroutine blocked at line 58 by acquiring the same `Mutex`.

## Kubernetes/30872

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#30872]|[pull request]|[patch]| Resource | AB-BA deadlock |

[kubernetes#30872]:(kubernetes30872_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/30872/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/30872

### Description

The lock is acquired both at lines 92 and 157.

## Kubernetes/38669

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#38669]|[pull request]|[patch]| Communication | Channel |

[kubernetes#38669]:(kubernetes38669_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/38669/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/38669

### Description

No sender for line 33.

## Kubernetes/5316

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#5316]|[pull request]|[patch]| Communication | Channel |

[kubernetes#5316]:(kubernetes5316_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/5316/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/5316

### Description

If the main goroutine selects a case that doesn’t consumes
the channels, the anonymous goroutine will be blocked on sending
to channel.

### Example execution

```go
G1                      G2
--------------------------------------
finishRequest()
.                       fn()
time.After()
.                       errCh<-/ch<-
--------------G2 leaks----------------
```

## Kubernetes/58107

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#58107]|[pull request]|[patch]| Resource | RWR deadlock |

[kubernetes#58107]:(kubernetes58107_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/58107/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/58107

### Description

The rules for read and write lock: allows concurrent read lock;
write lock has higher priority than read lock.

There are two queues (queue 1 and queue 2) involved in this bug,
and the two queues are protected by the same read-write lock
(`rq.workerLock.RLock()`). Before getting an element from queue 1 or
queue 2, `rq.workerLock.RLock()` is acquired. If the queue is empty,
`cond.Wait()` will be invoked. There is another goroutine (goroutine D),
which will periodically invoke `rq.workerLock.Lock()`. Under the following
situation, deadlock will happen. Queue 1 is empty, so that some goroutines
hold `rq.workerLock.RLock()`, and block at `cond.Wait()`. Goroutine D is
blocked when acquiring `rq.workerLock.Lock()`. Some goroutines try to process
jobs in queue 2, but they are blocked when acquiring `rq.workerLock.RLock()`,
since write lock has a higher priority.

The fix is to not acquire `rq.workerLock.RLock()`, while pulling data
from any queue. Therefore, when a goroutine is blocked at `cond.Wait()`,
`rq.workLock.RLock()` is not held.

### Example execution

```go
G3                      G4                      G5
--------------------------------------------------------------------
.                       .                       Sync()
rq.workerLock.RLock()   .                       .
q.cond.Wait()           .                       .
.                       .                       rq.workerLock.Lock()
.                       rq.workerLock.RLock()
.                       q.cond.L.Lock()
-----------------------------G3,G4,G5 leak-----------------------------
```

## Kubernetes/62464

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#62464]|[pull request]|[patch]| Resource | RWR deadlock |

[kubernetes#62464]:(kubernetes62464_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/62464/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/62464

### Description

This is another example for recursive read lock bug. It has
been noticed by the go developers that RLock should not be
recursively used in the same thread.

### Example execution

```go
G1                                  G2
--------------------------------------------------------
m.reconcileState()
m.state.GetCPUSetOrDefault()
s.RLock()
s.GetCPUSet()
.                                   p.RemoveContainer()
.                                   s.GetDefaultCPUSet()
.                                   s.SetDefaultCPUSet()
.                                   s.Lock()
s.RLock()
---------------------G1,G2 leak--------------------------
```

## Kubernetes/6632

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#6632]|[pull request]|[patch]| Mixed | Channel & Lock |

[kubernetes#6632]:(kubernetes6632_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/6632/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/6632

### Description

When `resetChan` is full, `WriteFrame` holds the lock and blocks
on the channel. Then `monitor()` fails to close the `resetChan`
because the lock is already held by `WriteFrame`.


### Example execution

```go
G1                      G2                  helper goroutine
----------------------------------------------------------------
i.monitor()
<-i.conn.closeChan
.                       i.WriteFrame()
.                       i.writeLock.Lock()
.                       i.resetChan <-
.                       .                   i.conn.closeChan<-
i.writeLock.Lock()
----------------------G1,G2 leak--------------------------------
```

## Kubernetes/70277

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[kubernetes#70277]|[pull request]|[patch]| Communication | Channel |

[kubernetes#70277]:kubernetes70277_test.go
[patch]:https://github.com/kubernetes/kubernetes/pull/70277/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/70277

### Description

`wait.poller()` returns a function with type `WaitFunc`.
the function creates a goroutine and the goroutine only
quits when after or done closed.

The `doneCh` defined at line 70 is never closed.

## Moby/17176

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#17176]|[pull request]|[patch]| Resource | Double locking |

[moby#17176]:(moby17176_test.go)
[patch]:https://github.com/moby/moby/pull/17176/files
[pull request]:https://github.com/moby/moby/pull/17176

### Description

`devices.nrDeletedDevices` takes `devices.Lock()` but does
not release it (line 36) if there are no deleted devices. This will block
other goroutines trying to acquire `devices.Lock()`.

## Moby/21233

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#21233]|[pull request]|[patch]| Communication | Channel |

[moby#21233]:(moby21233_test.go)
[patch]:https://github.com/moby/moby/pull/21233/files
[pull request]:https://github.com/moby/moby/pull/21233

### Description

This test was checking that it received every progress update that was
produced. But delivery of these intermediate progress updates is not
guaranteed. A new update can overwrite the previous one if the previous
one hasn't been sent to the channel yet.

The call to `t.Fatalf` terminated the current goroutine which was consuming
the channel, which caused a deadlock and eventual test timeout rather
than a proper failure message.

### Example execution

```go
G1                      G2                  G3
----------------------------------------------------------
testTransfer()          .                   .
tm.Transfer()           .                   .
t.Watch()               .                   .
.                       WriteProgress()     .
.                       ProgressChan<-      .
.                       .                   <-progressChan
.                       ...                 ...
.                       return              .
.                                           <-progressChan
<-watcher.running
----------------------G1,G3 leak--------------------------
```

## Moby/25384

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#25384]|[pull request]|[patch]| Mixed | Misuse WaitGroup |

[moby#25384]:(moby25384_test.go)
[patch]:https://github.com/moby/moby/pull/25384/files
[pull request]:https://github.com/moby/moby/pull/25384

### Description

When `n=1` (where `n` is `len(pm.plugins)`), the location of `group.Wait()` doesn’t matter.
When `n > 1`, `group.Wait()` is invoked in each iteration. Whenever
`group.Wait()` is invoked, it waits for `group.Done()` to be executed `n` times.
However, `group.Done()` is only executed once in one iteration.

Misuse of sync.WaitGroup

## Moby/27782

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#27782]|[pull request]|[patch]| Communication | Channel & Condition Variable |

[moby#27782]:(moby27782_test.go)
[patch]:https://github.com/moby/moby/pull/27782/files
[pull request]:https://github.com/moby/moby/pull/27782

### Description

### Example execution

```go
G1                      G2                         G3
-----------------------------------------------------------------------
InitializeStdio()
startLogging()
l.ReadLogs()
NewLogWatcher()
.                       l.readLogs()
container.Reset()       .
LogDriver.Close()       .
r.Close()               .
close(w.closeNotifier)  .
.                       followLogs(logWatcher)
.                       watchFile()
.                       New()
.                       NewEventWatcher()
.                       NewWatcher()
.                       .                       	w.readEvents()
.                       .                       	event.ignoreLinux()
.                       .                       	return false
.                       <-logWatcher.WatchClose()   .
.                       fileWatcher.Remove()        .
.                       w.cv.Wait()                 .
.                       .                       	w.Events <- event
------------------------------G2,G3 leak-------------------------------
```

## Moby/28462

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#28462]|[pull request]|[patch]| Mixed | Channel & Lock |

[moby#28462]:(moby28462_test.go)
[patch]:https://github.com/moby/moby/pull/28462/files
[pull request]:https://github.com/moby/moby/pull/28462

### Description

One goroutine may acquire a lock and try to send a message over channel `stop`,
while the other will try to acquire the same lock. With the wrong ordering,
both goroutines will leak.

### Example execution

```go
G1                        	G2
--------------------------------------------------------------
monitor()
handleProbeResult()
.                       	d.StateChanged()
.                       	c.Lock()
.                       	d.updateHealthMonitorElseBranch()
.                       	h.CloseMonitorChannel()
.                       	s.stop <- struct{}{}
c.Lock()
----------------------G1,G2 leak------------------------------
```

## Moby/30408

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#30408]|[pull request]|[patch]| Communication | Condition Variable |

[moby#30408]:(moby30408_test.go)
[patch]:https://github.com/moby/moby/pull/30408/files
[pull request]:https://github.com/moby/moby/pull/30408

### Description

`Wait()` at line 22 has no corresponding `Signal()` or `Broadcast()`.

### Example execution

```go
G1                 G2
------------------------------------------
testActive()
.                  p.waitActive()
.                  p.activateWait.L.Lock()
.                  p.activateWait.Wait()
<-done
-----------------G1,G2 leak---------------
```

## Moby/33781

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#33781]|[pull request]|[patch]| Communication | Channel & Context |

[moby#33781]:(moby33781_test.go)
[patch]:https://github.com/moby/moby/pull/33781/files
[pull request]:https://github.com/moby/moby/pull/33781

### Description

The goroutine created using an anonymous function is blocked 
sending a message over an unbuffered channel. However there
exists a path in the parent goroutine where the parent function
will return without draining the channel.

### Example execution

```go
G1             	G2             G3
----------------------------------------
monitor()       .
<-time.After()  .
.           	.
<-stop          stop<-
.
cancelProbe()
return
.                               result<-
----------------G3 leak------------------
```

## Moby/36114

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#36114]|[pull request]|[patch]| Resource | Double locking |

[moby#36114]:(moby36114_test.go)
[patch]:https://github.com/moby/moby/pull/36114/files
[pull request]:https://github.com/moby/moby/pull/36114

### Description

The the lock for the struct svm has already been locked when calling
`svm.hotRemoveVHDsAtStart()`.

## Moby/4951

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#4951]|[pull request]|[patch]| Resource | AB-BA deadlock |

[moby#4951]:(moby4951_test.go)
[patch]:https://github.com/moby/moby/pull/4951/files
[pull request]:https://github.com/moby/moby/pull/4951

### Description

The root cause and patch is clearly explained in the commit
description. The global lock is `devices.Lock()`, and the device
lock is `baseInfo.lock.Lock()`. It is very likely that this bug
can be reproduced.

## Moby/7559

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[moby#7559]|[pull request]|[patch]| Resource | Double locking |

[moby#7559]:(moby7559_test.go)
[patch]:https://github.com/moby/moby/pull/7559/files
[pull request]:https://github.com/moby/moby/pull/7559

### Description

Line 25 is missing a call to `.Unlock`.

### Example execution

```go
G1
---------------------------
proxy.connTrackLock.Lock()
if err != nil { continue }
proxy.connTrackLock.Lock()
-----------G1 leaks--------
```

## Serving/2137

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[serving#2137]|[pull request]|[patch]| Mixed | Channel & Lock |

[serving#2137]:(serving2137_test.go)
[patch]:https://github.com/ knative/serving/pull/2137/files
[pull request]:https://github.com/ knative/serving/pull/2137

### Description

### Example execution

```go
G1                           	G2                      	G3
----------------------------------------------------------------------------------
b.concurrentRequests(2)         .                           .
b.concurrentRequest()           .                           .
r.lock.Lock()                   .                           .
.                               start.Done()                .
start.Wait()                    .                           .
b.concurrentRequest()           .                           .
r.lock.Lock()                   .                           .
.                               .                           start.Done()
start.Wait()                    .                           .
unlockAll(locks)                .                           .
unlock(lc)                      .                           .
req.lock.Unlock()               .                           .
ok := <-req.accepted            .                           .
.                            	b.Maybe()                   .
.                            	b.activeRequests <- t       .
.                            	thunk()                     .
.                            	r.lock.Lock()               .
.                            	.                           b.Maybe()
.                            	.                           b.activeRequests <- t
----------------------------G1,G2,G3 leak-----------------------------------------
```

## Syncthing/4829

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[syncthing#4829]|[pull request]|[patch]| Resource | Double locking |

[syncthing#4829]:(syncthing4829_test.go)
[patch]:https://github.com/syncthing/syncthing/pull/4829/files
[pull request]:https://github.com/syncthing/syncthing/pull/4829

### Description

Double locking at line 17 and line 30.

### Example execution

```go
G1
---------------------------
mapping.clearAddresses()
m.mut.Lock() [L2]
m.notify(...)
m.mut.RLock() [L2]
----------G1 leaks---------
```

## Syncthing/5795

| Bug ID |  Ref | Patch | Type | Sub-type |
| ---- | ---- | ---- | ---- | ---- |
|[syncthing#5795]|[pull request]|[patch]| Communication | Channel |

[syncthing#5795]:(syncthing5795_test.go)
[patch]:https://github.com/syncthing/syncthing/pull/5795/files
[pull request]:https://github.com/syncthing/syncthing/pull/5795

### Description

`<-c.dispatcherLoopStopped` at line 82 is blocking forever because
`dispatcherLoop()` is blocking at line 72.

### Example execution

```go
G1                            G2
--------------------------------------------------------------
c.Start()
go c.dispatcherLoop() [G3]
.                             select [<-c.inbox, <-c.closed]
c.inbox <- <================> [<-c.inbox]
<-c.dispatcherLoopStopped     .
.                             default
.                             c.ccFn()/c.Close()
.                             close(c.closed)
.                             <-c.dispatcherLoopStopped
---------------------G1,G2 leak-------------------------------
```
