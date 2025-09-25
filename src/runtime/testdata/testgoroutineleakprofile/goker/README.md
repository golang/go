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

The examples have been modified in order to run the goroutine leak
profiler. Buggy snippet are not located inside a unit test, but are
instead each moved to its separate application.

Each concurrency bug is independently executed within its own copy
of the program by the Go runtime, and then in separate child goroutines.
The main goroutine only sets up the child goroutines in an attempt to
reproduce the bug, followed by a short delay (from 1ms to
several seconds), and then a request for a goroutine leak profile.

The profile is analyzed to ensure that no unexpected leaks occurred,
and that the expected leaks did occur (except if the leak is flaky,
in which case the only purpose of the expected leak list is to protect
against unexpected leaks)

The entries below document each of the corresponding leaks.

## Cockroach/10214

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#10214]|[pull request]|[patch]| Blocking | Resource Deadlock | AB-BA deadlock |

[cockroach#10214]:(cockroach10214_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/10214/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/10214
 
### Description

This is some description from previous researchers

> This deadlock is caused by different order when acquiring
> coalescedMu.Lock() and raftMu.Lock(). The fix is to refactor sendQueuedHeartbeats()
> so that cockroachdb can unlock coalescedMu before locking raftMu.

### Backtrace

```
goroutine 25176 [semacquire]:
sync.runtime_SemacquireMutex(0xc00009a7b4, 0x0, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc00009a7b0)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*Replica).maybeCoalesceHeartbeat(0xc00000c320, 0xc00000c4e0)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:74 +0x99
command-line-arguments.(*Replica).maybeQuiesceLocked(0xc00000c320, 0xc00001b8f0)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:64 +0x43
command-line-arguments.(*Replica).tickRaftMuLocked(0xc00000c320)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:58 +0x69
command-line-arguments.(*Replica).tick(0xc00000c320)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:51 +0x64
command-line-arguments.TestCockroach10214.func2(0xc00000c320)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:103 +0x2b
created by command-line-arguments.TestCockroach10214
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:102 +0x217

 Goroutine 25175 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 25175 [semacquire]:
sync.runtime_SemacquireMutex(0xc00000c324, 0x0, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc00000c320)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*Replica).reportUnreachable(0xc00000c320)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:42 +0x78
command-line-arguments.(*Store).sendQueuedHeartbeatsToNode(0xc00009a7b0)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:31 +0x56
command-line-arguments.(*Store).sendQueuedHeartbeats(0xc00009a7b0)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:24 +0x6d
command-line-arguments.TestCockroach10214.func1(0xc00009a7b0)
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:99 +0x2b
created by command-line-arguments.TestCockroach10214
    /root/gobench/gobench/goker/blocking/cockroach/10214/cockroach10214_test.go:98 +0x1f5
```

## Cockroach/1055

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#1055]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & WaitGroup |

[cockroach#1055]:(cockroach1055_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/1055/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/1055
 
### Description

This is some description from developers

> 1. Stop() is called and blocked at s.stop.Wait() after acquiring the lock.
> 2. StartTask() is called and attempts to acquire the lock. It is then blocked.
> 3. Stop() never finishes since the task doesn't call SetStopped.

### Backtrace

```
goroutine 16 [semacquire]:
sync.runtime_Semacquire(0xc00001a778)
	/usr/local/go/src/runtime/sema.go:56 +0x42
sync.(*WaitGroup).Wait(0xc00001a770)
	/usr/local/go/src/sync/waitgroup.go:130 +0x64
command-line-arguments.(*Stopper).Stop(0xc00001a750)
	/root/gobench/gobench/goker/blocking/cockroach/1055/cockroach1055_test.go:46 +0x70
command-line-arguments.TestCockroach1055.func2(0xc00000c0c0, 0x3, 0x4, 0xc0000628a0)
	/root/gobench/gobench/goker/blocking/cockroach/1055/cockroach1055_test.go:89 +0x69
created by command-line-arguments.TestCockroach1055
	/root/gobench/gobench/goker/blocking/cockroach/1055/cockroach1055_test.go:84 +0x29c

goroutine 15 [chan receive]:
command-line-arguments.TestCockroach1055.func1(0xc00001a750)
	/root/gobench/gobench/goker/blocking/cockroach/1055/cockroach1055_test.go:78 +0x4a
created by command-line-arguments.TestCockroach1055
	/root/gobench/gobench/goker/blocking/cockroach/1055/cockroach1055_test.go:76 +0x21c
```

## Cockroach/10790

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#10790]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[cockroach#10790]:(cockroach10790_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/10790/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/10790
 
### Description

This is some description from previous researchers

> It is possible that a message from ctxDone will make the function beginCmds
> returns without draining the channel ch, so that goroutines created by anonymous
> function will leak.

Possible intervening

```
///
/// G1					G2				helper goroutine
/// 									r.sendChans()
/// r.beginCmds()
/// 									ch1 <- true
/// <- ch1
///										ch2 <- true
///	...					...				...
///						cancel()
///	<- ch1
///	------------------G1 leak--------------------------
```

### Backtrace

```
goroutine 603 [chan receive]:
command-line-arguments.(*Replica).beginCmds.func1(0xc00000c5a0)
    /root/gobench/gobench/goker/blocking/cockroach/10790/cockroach10790_test.go:51 +0x52
created by command-line-arguments.(*Replica).beginCmds
    /root/gobench/gobench/goker/blocking/cockroach/10790/cockroach10790_test.go:49 +0x13f

 Goroutine 604 in state chan receive, with command-line-arguments.(*Replica).beginCmds.func1 on top of the stack:
goroutine 604 [chan receive]:
command-line-arguments.(*Replica).beginCmds.func1(0xc00000c5a0)
    /root/gobench/gobench/goker/blocking/cockroach/10790/cockroach10790_test.go:51 +0x52
created by command-line-arguments.(*Replica).beginCmds
    /root/gobench/gobench/goker/blocking/cockroach/10790/cockroach10790_test.go:49 +0x13f
```

## Cockroach/13197

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#13197]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[cockroach#13197]:(cockroach13197_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/13197/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/13197
 
### Description

This is some description from previous researchers

> One goroutine executing (*Tx).awaitDone() blocks and
> waiting for a signal context.Done().

Possible intervening

```
/// G1 				G2
/// begin()
/// 				awaitDone()
/// 				<-tx.ctx.Done()
/// return
/// -----------G2 leak-------------
```

### Backtrace

```
goroutine 19 [chan receive]:
command-line-arguments.(*Tx).awaitDone(0xc000130040)
    /root/gobench/gobench/goker/blocking/cockroach/13197/cockroach13197_test.go:27 +0x4b
created by command-line-arguments.(*DB).begin
    /root/gobench/gobench/goker/blocking/cockroach/13197/cockroach13197_test.go:17 +0xba
```

## Cockroach/13755

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#13755]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[cockroach#13755]:(cockroach13755_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/13755/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/13755
 
### Description

This is some description from previous researchers

> The buggy code does not close the db query result (rows),
> so that one goroutine running (*Rows).awaitDone is blocked forever.
> The blocking goroutine is waiting for cancel signal from context.

Possible intervening

```
/// G1 						G2
/// initContextClose()
/// 						awaitDone()
/// 						<-tx.ctx.Done()
/// return
/// ---------------G2 leak-----------------
```

### Backtrace

```
goroutine 19 [chan receive]:
command-line-arguments.(*Rows).awaitDone(0xc000102028, 0x5766e0, 0xc000108600)
    /root/gobench/gobench/goker/blocking/cockroach/13755/cockroach13755_test.go:19 +0x48
created by command-line-arguments.(*Rows).initContextClose
    /root/gobench/gobench/goker/blocking/cockroach/13755/cockroach13755_test.go:15 +0x82
```

## Cockroach/1462

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#1462]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & WaitGroup |

[cockroach#1462]:(cockroach1462_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/1462/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/1462
 
### Description

`s.stop.Wait()` in `Stop()` not guaranteed to be invoked

### Backtrace

```
goroutine 12 [chan receive]:
goroutine 4613 [semacquire, 9 minutes]:
sync.runtime_Semacquire(0xc0000ea418)
	/usr/local/go/src/runtime/sema.go:56 +0x42
sync.(*WaitGroup).Wait(0xc0000ea410)
	/usr/local/go/src/sync/waitgroup.go:130 +0x64
command-line-arguments.(*Stopper).Stop(0xc0000ea400)
	/root/gobench/gobench/goker/blocking/cockroach/1462/cockroach1462_test.go:79 +0x5f
command-line-arguments.TestCockroach1462(0xc0000e7320)
	/root/gobench/gobench/goker/blocking/cockroach/1462/cockroach1462_test.go:139 +0x1fc
testing.tRunner(0xc0000e7320, 0x554420)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b

goroutine 4614 [chan send, 9 minutes]:
command-line-arguments.(*localInterceptableTransport).start.func1()
	/root/gobench/gobench/goker/blocking/cockroach/1462/cockroach1462_test.go:115 +0x46
command-line-arguments.(*Stopper).RunWorker.func1(0xc0000ea400, 0xc0000e42c0)
	/root/gobench/gobench/goker/blocking/cockroach/1462/cockroach1462_test.go:31 +0x4f
created by command-line-arguments.(*Stopper).RunWorker
	/root/gobench/gobench/goker/blocking/cockroach/1462/cockroach1462_test.go:29 +0x67
```

## Cockroach/16167

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#16167]|[pull request]|[patch]| Blocking | Resource Deadlock | Double Locking |

[cockroach#16167]:(cockroach16167_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/16167/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/16167
 
### Description

This is some description from previous researchers

> This is another example for deadlock caused by recursively
> acquiring RWLock. There are two lock variables (systemConfigCond and systemConfigMu)
> involved in this bug, but they are actually the same lock, which can be found from
> the following code.
> There are two goroutine involved in this deadlock. The first goroutine acquires
> systemConfigMu.Lock() firstly, then tries to acquire systemConfigMu.RLock(). The
> second goroutine tries to acquire systemConfigMu.Lock(). If the second goroutine
> interleaves in between the two lock operations of the first goroutine, deadlock will happen.

Possible intervening

```
/// G1 							G2
/// e.Start()
/// e.updateSystemConfig()
/// 							e.execParsed()
/// 							e.systemConfigCond.L.Lock()
/// e.systemConfigMu.Lock()
/// 							e.systemConfigMu.RLock()
/// ----------------------G1,G2 deadlock--------------------
```

### Backtrace

```
goroutine 44794 [semacquire]:
sync.runtime_SemacquireMutex(0xc0003f7874, 0x7fdde274ae00, 0x0)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).RLock(...)
	/usr/local/go/src/sync/rwmutex.go:50
command-line-arguments.(*Executor).getDatabaseCache(0xc0003f7860)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:69 +0x91
command-line-arguments.(*Session).resetForBatch(...)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:38
command-line-arguments.(*Executor).Prepare(...)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:65
command-line-arguments.PreparedStatements.New(...)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:30
command-line-arguments.(*Executor).execStmtInOpenTxn(...)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:61
command-line-arguments.(*Executor).execStmtsInCurrentTxn(...)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:57
command-line-arguments.runTxnAttempt(0xc0003f7860, 0xc0000aae28)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:79 +0x35
command-line-arguments.(*Executor).execParsed(0xc0003f7860, 0xc0000aae28)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:53 +0x7e
command-line-arguments.TestCockroach16167(0xc000482d80)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:101 +0xd2
testing.tRunner(0xc000482d80, 0x54d0d8)
	/usr/local/go/src/testing/testing.go:1123 +0xef
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1168 +0x2b3

goroutine 44795 [semacquire]:
sync.runtime_SemacquireMutex(0xc0003f7870, 0x0, 0x0)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).Lock(0xc0003f7868)
	/usr/local/go/src/sync/rwmutex.go:103 +0x85
command-line-arguments.(*Executor).updateSystemConfig(0xc0003f7860)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:74 +0x45
command-line-arguments.(*Executor).Start(0xc0003f7860)
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:47 +0x2b
created by command-line-arguments.TestCockroach16167
	/home/yuanting/work-gobench/gobench/gobench/goker/blocking/cockroach/16167/cockroach16167_test.go:100 +0xba
```

## Cockroach/18101

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#18101]|[pull request]|[patch]| Blocking | Resource Deadlock | Double Locking |

[cockroach#18101]:(cockroach18101_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/18101/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/18101
 
### Description

This is some description from previous researchers

> context.Done() signal only stops the goroutine who pulls data
> from a channel, while does not stops goroutines which send data
> to the channel. This causes all goroutines trying to send data
> through the channel to block.

Possible intervening

```
///
/// G1					G2					helper goroutine
/// restore()
/// 					splitAndScatter()
/// <-readyForImportCh
/// 					readyForImportCh<-
/// ...					...
/// 										cancel()
/// return
/// 					readyForImportCh<-
/// -----------------------G2 leak-------------------------
```

### Backtrace

```
goroutine 33 [chan send]:
command-line-arguments.splitAndScatter(0x576500, 0xc000078600, 0xc000180000)
    /root/gobench/gobench/goker/blocking/cockroach/18101/cockroach18101_test.go:28 +0x4b
command-line-arguments.restore.func1(0xc000180000, 0x576500, 0xc000078600)
    /root/gobench/gobench/goker/blocking/cockroach/18101/cockroach18101_test.go:15 +0x62
created by command-line-arguments.restore
    /root/gobench/gobench/goker/blocking/cockroach/18101/cockroach18101_test.go:13 +0x70
```

## Cockroach/2448

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#2448]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[cockroach#2448]:(cockroach2448_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/2448/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/2448
 
### Description

This is some description from previous researchers

>  This bug is caused by two goroutines waiting for each other
>  to unblock their channels.
>
>  (1) MultiRaft sends the commit event for the Membership change
>
>  (2) store.processRaft takes it and begins processing
>
>  (3) another command commits and triggers another sendEvent, but
  	   this blocks since store.processRaft isn't ready for another
  	   select. Consequently the main MultiRaft loop is waiting for
  	   that as well.
>
>  (4) the Membership change was applied to the range, and the store
       now tries to execute the callback
>
>  (5) the callback tries to write to callbackChan, but that is
  	   consumed by the MultiRaft loop, which is currently waiting
  	   for store.processRaft to consume from the events channel,
  	   which it will only do after the callback has completed.

Possible intervening
```
G1								G2
s.processRaft()
e := <-s.multiraft.Events
								st.start()
 								s.handleWriteResponse()
 								s.processCommittedEntry()
 								s.sendEvent()
 								m.Events <- event
								...
 								s.handleWriteResponse()
 								s.processCommittedEntry()
 								s.sendEvent()
 								m.Events <- event
callback()
s.callbackChan <- func()
```

### Backtrace

```
goroutine 19 [select]:
command-line-arguments.(*state).processCommittedEntry.func1()
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:57 +0xbc
command-line-arguments.(*Store).processRaft(0xc0000ae038)
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:80 +0xfc
created by command-line-arguments.TestCockroach2448
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:106 +0x1d7

 Goroutine 20 in state select, with command-line-arguments.(*MultiRaft).sendEvent on top of the stack:
goroutine 20 [select]:
command-line-arguments.(*MultiRaft).sendEvent(0xc0000be040, 0x50fc80, 0xc00000e010)
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:28 +0xc6
command-line-arguments.(*state).processCommittedEntry(0xc0000ae030)
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:55 +0x91
command-line-arguments.(*state).handleWriteResponse(...)
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:51
command-line-arguments.(*state).start(0xc0000ae030)
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:46 +0xfe
created by command-line-arguments.TestCockroach2448
    /root/gobench/gobench/goker/blocking/cockroach/2448/cockroach2448_test.go:107 +0x1f9
```

## Cockroach/24808

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#24808]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[cockroach#24808]:(cockroach24808_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/24808/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/24808
 
### Description

This is some description from developers

> When we Start the Compactor, it may already have received
> Suggestions, deadlocking the previously blocking write to a full
> channel.

### Backtrace

```
goroutine 33 [chan send]:
command-line-arguments.(*Compactor).Start(0xc0000ce010, 0x574b00, 0xc0000aa010, 0xc0000b4040)
	/root/gobench/gobench/goker/blocking/cockroach/24808/cockroach24808_test.go:50 +0x3c
command-line-arguments.TestCockroach24808(0xc0000d0120)
	/root/gobench/gobench/goker/blocking/cockroach/24808/cockroach24808_test.go:70 +0x18f
testing.tRunner(0xc0000d0120, 0x552e90)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b
```

## Cockroach/25456

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#25456]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[cockroach#25456]:(cockroach25456_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/25456/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/25456
 
### Description

This is some description from developers

> When CheckConsistency returns an error, the queue checks whether the
  store is draining to decide whether the error is worth logging.
  Unfortunately this check was incorrect and would block until the store
  actually started draining.

This bug is because of channel communication mismatch

### Backtrace

```
goroutine 6 [chan receive]:
command-line-arguments.(*consistencyQueue).process(...)
	/root/gobench/gobench/goker/blocking/cockroach/25456/cockroach25456_test.go:51
command-line-arguments.TestCockroach25456(0xc00008e120)
	/root/gobench/gobench/goker/blocking/cockroach/25456/cockroach25456_test.go:77 +0x16c
testing.tRunner(0xc00008e120, 0x550718)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b
```

## Cockroach/35073

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#35073]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[cockroach#35073]:(cockroach35073_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/35073/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/35073
 
### Description

This is some description from developers

> Previously, the outbox could fail during startup without closing its
  RowChannel. This could lead to deadlocked flows in rare cases.

Channel communication mismatch

### Backtrace

```
goroutine 18 [chan send]:
command-line-arguments.(*RowChannel).Push(...)
	/root/gobench/gobench/goker/blocking/cockroach/35073/cockroach35073_test.go:49
command-line-arguments.TestCockroach35073(0xc000144120)
	/root/gobench/gobench/goker/blocking/cockroach/35073/cockroach35073_test.go:110 +0x222
testing.tRunner(0xc000144120, 0x5519d8)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b

goroutine 19 [chan send]:
command-line-arguments.(*RowChannel).Push(...)
	/root/gobench/gobench/goker/blocking/cockroach/35073/cockroach35073_test.go:49
command-line-arguments.TestCockroach35073.func1(0xc000104480, 0xc0001180a0)
	/root/gobench/gobench/goker/blocking/cockroach/35073/cockroach35073_test.go:103 +0x62
created by command-line-arguments.TestCockroach35073
	/root/gobench/gobench/goker/blocking/cockroach/35073/cockroach35073_test.go:102 +0x18c
```

## Cockroach/35931

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#35931]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[cockroach#35931]:(cockroach35931_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/35931/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/35931
 
### Description

This is some description from developers

> Previously, if a processor that reads from multiple inputs was waiting
  on one input to provide more data, and the other input was full, and
  both inputs were connected to inbound streams, it was possible to
  deadlock the system during flow cancellation when trying to propagate
  the cancellation metadata messages into the flow. The problem was that
  the cancellation method wrote metadata messages to each inbound stream
  one at a time, so if the first one was full, the canceller would block
  and never send a cancellation message to the second stream, which was
  the one actually being read from.

Channel mismatch

### Backtrace

```
goroutine 6 [chan send]:
command-line-arguments.(*RowChannel).Push(0xc00000e038)
	/root/gobench/gobench/goker/blocking/cockroach/35931/cockroach35931_test.go:22 +0x38
command-line-arguments.(*Flow).cancel(0xc00000c080)
	/root/gobench/gobench/goker/blocking/cockroach/35931/cockroach35931_test.go:69 +0xa9
command-line-arguments.TestCockroach35931(0xc00010e120)
	/root/gobench/gobench/goker/blocking/cockroach/35931/cockroach35931_test.go:113 +0x322
testing.tRunner(0xc00010e120, 0x552258)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b
```

## Cockroach/3710

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#3710]|[pull request]|[patch]| Blocking | Resource Deadlock | RWR Deadlock |

[cockroach#3710]:(cockroach3710_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/3710/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/3710
 
### Description

This is some description from previous researchers

> This deadlock is casued by acquiring a RLock twice in a call chain.
> ForceRaftLogScanAndProcess(acquire s.mu.RLock()) ->MaybeAdd()->shouldQueue()->
> getTruncatableIndexes()->RaftStatus(acquire s.mu.Rlock())

Possible intervening

```
/// G1 										G2
/// store.ForceRaftLogScanAndProcess()
/// s.mu.RLock()
/// s.raftLogQueue.MaybeAdd()
/// bq.impl.shouldQueue()
/// getTruncatableIndexes()
/// r.store.RaftStatus()
/// 										store.processRaft()
/// 										s.mu.Lock()
/// s.mu.RLock()
/// ----------------------G1,G2 deadlock---------------------
```

### Backtrace

```
goroutine 12205 [semacquire]:
sync.runtime_SemacquireMutex(0xc00009bbdc, 0x0, 0x0)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).RLock(...)
    /usr/local/go/src/sync/rwmutex.go:50
command-line-arguments.(*Store).RaftStatus(0xc00009bbc0)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:28 +0x92
command-line-arguments.getTruncatableIndexes(...)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:68
command-line-arguments.(*raftLogQueue).shouldQueue(...)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:64
command-line-arguments.(*baseQueue).MaybeAdd(0xc0000965f0, 0xc0004c0058)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:58 +0x6e
command-line-arguments.(*Store).ForceRaftLogScanAndProcess(0xc00009bbc0)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:22 +0xa2
created by command-line-arguments.TestCockroach3710
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:93 +0x9b

 Goroutine 12036 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 12036 [semacquire]:
sync.runtime_SemacquireMutex(0xc00009bbd8, 0xc00009b000, 0x0)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).Lock(0xc00009bbd0)
    /usr/local/go/src/sync/rwmutex.go:103 +0x88
command-line-arguments.(*Store).processRaft.func1(0xc00009bbc0)
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:36 +0x4b
created by command-line-arguments.(*Store).processRaft
    /root/gobench/gobench/goker/blocking/cockroach/3710/cockroach3710_test.go:33 +0x3f
```

## Cockroach/584

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#584]|[pull request]|[patch]| Blocking | Resource Deadlock | Double Locking |

[cockroach#584]:(cockroach584_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/584/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/584
 
### Description

This is some description from developers

> I'm guessing some of the goroutines might get into deadlock during shutdown. 
> (We cannot use defer as Lock is called inside a for loop.)

Missing unlock before break the loop

### Backtrace

```
goroutine 7 [semacquire]:
sync.runtime_SemacquireMutex(0xc0000140f4, 0x0, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc0000140f0)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*Gossip).manage(0xc0000140f0)
    /root/gobench/gobench/goker/blocking/cockroach/584/cockroach584_test.go:27 +0x6c
command-line-arguments.TestCockroach584.func1(0xc0000140f0)
    /root/gobench/gobench/goker/blocking/cockroach/584/cockroach584_test.go:40 +0x39
created by command-line-arguments.TestCockroach584
    /root/gobench/gobench/goker/blocking/cockroach/584/cockroach584_test.go:36 +0xa6
```

## Cockroach/6181

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#6181]|[pull request]|[patch]| Blocking | Resource Deadlock | RWR Deadlock |

[cockroach#6181]:(cockroach6181_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/6181/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/6181
 
### Description

Possible intervening

```
/// G1 									G2							G3					...
/// testRangeCacheCoalescedRquests()
/// initTestDescriptorDB()
/// pauseLookupResumeAndAssert()
/// return
/// 									doLookupWithToken()
///																 	doLookupWithToken()
///										rc.LookupRangeDescriptor()
///																	rc.LookupRangeDescriptor()
///										rdc.rangeCacheMu.RLock()
///										rdc.String()
///																	rdc.rangeCacheMu.RLock()
///																	fmt.Printf()
///																	rdc.rangeCacheMu.RUnlock()
///																	rdc.rangeCacheMu.Lock()
///										rdc.rangeCacheMu.RLock()
/// -------------------------------------G2,G3,... deadlock--------------------------------------
```

### Backtrace

```
goroutine 34 [semacquire]:
sync.runtime_Semacquire(0xc000014068)
    /usr/local/go/src/runtime/sema.go:56 +0x42
sync.(*WaitGroup).Wait(0xc000014060)
    /usr/local/go/src/sync/waitgroup.go:130 +0x64
command-line-arguments.testRangeCacheCoalescedRquests.func1()
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:55 +0xa8
command-line-arguments.testRangeCacheCoalescedRquests()
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:57 +0x86
created by command-line-arguments.TestCockroach6181
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:62 +0x88

 Goroutine 5 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 5 [semacquire]:
sync.runtime_SemacquireMutex(0xc00001806c, 0x40a000, 0x0)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).RLock(...)
    /usr/local/go/src/sync/rwmutex.go:50
command-line-arguments.(*rangeDescriptorCache).String(0xc000018060, 0x0, 0x0)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:31 +0xb0
fmt.(*pp).handleMethods(0xc00010a000, 0x7f8d00000073, 0xc000036601)
    /usr/local/go/src/fmt/print.go:630 +0x302
fmt.(*pp).printArg(0xc00010a000, 0x522e40, 0xc000018060, 0x73)
    /usr/local/go/src/fmt/print.go:713 +0x1e4
fmt.(*pp).doPrintf(0xc00010a000, 0x54c53e, 0x1b, 0xc000110f90, 0x1, 0x1)
    /usr/local/go/src/fmt/print.go:1030 +0x15a
fmt.Fprintf(0x571d40, 0xc0000ac000, 0x54c53e, 0x1b, 0xc000036790, 0x1, 0x1, 0x0, 0x0, 0x0)
    /usr/local/go/src/fmt/print.go:204 +0x72
fmt.Printf(...)
    /usr/local/go/src/fmt/print.go:213
command-line-arguments.(*rangeDescriptorCache).LookupRangeDescriptor(0xc000018060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:24 +0xa0
command-line-arguments.doLookupWithToken(...)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:41
command-line-arguments.testRangeCacheCoalescedRquests.func1.1(0xc00000e010, 0xc000014060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:51 +0x2e
created by command-line-arguments.testRangeCacheCoalescedRquests.func1
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:50 +0x8b

 Goroutine 6 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 6 [semacquire]:
sync.runtime_SemacquireMutex(0xc00001806c, 0x0, 0x0)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).RLock(...)
    /usr/local/go/src/sync/rwmutex.go:50
command-line-arguments.(*rangeDescriptorCache).LookupRangeDescriptor(0xc000018060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:23 +0x106
command-line-arguments.doLookupWithToken(...)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:41
command-line-arguments.testRangeCacheCoalescedRquests.func1.1(0xc00000e010, 0xc000014060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:51 +0x2e
created by command-line-arguments.testRangeCacheCoalescedRquests.func1
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:50 +0x8b

 Goroutine 7 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 7 [semacquire]:
sync.runtime_SemacquireMutex(0xc000018068, 0xc000064000, 0x0)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).Lock(0xc000018060)
    /usr/local/go/src/sync/rwmutex.go:103 +0x88
command-line-arguments.(*rangeDescriptorCache).LookupRangeDescriptor(0xc000018060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:26 +0xbf
command-line-arguments.doLookupWithToken(...)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:41
command-line-arguments.testRangeCacheCoalescedRquests.func1.1(0xc00000e010, 0xc000014060)
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:51 +0x2e
created by command-line-arguments.testRangeCacheCoalescedRquests.func1
    /root/gobench/gobench/goker/blocking/cockroach/6181/cockroach6181_test.go:50 +0x8b
```

## Cockroach/7504

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#7504]|[pull request]|[patch]| Blocking | Resource Deadlock | AB-BA Deadlock |

[cockroach#7504]:(cockroach7504_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/7504/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/7504
 
### Description

This is some description from previous researchers

> There are locking leaseState, tableNameCache in Release(), but
> tableNameCache,LeaseState in AcquireByName.  It is AB and BA deadlock.

### Backtrace

```
goroutine 49274 [semacquire]:
sync.runtime_SemacquireMutex(0xc000102814, 0x0, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000102810)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*tableNameCache).remove(0xc000102810, 0xc000014510)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:82 +0x112
command-line-arguments.(*tableState).removeLease(0xc00000c3a0, 0xc000014510)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:57 +0x54
command-line-arguments.(*tableState).release(0xc00000c3a0, 0xc000014510)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:53 +0xb6
command-line-arguments.(*LeaseManager).Release(0xc000060a80, 0xc000014510)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:116 +0x70
command-line-arguments.TestCockroach7504.func2(0xc000060a80, 0xc00000c380)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:161 +0x44
created by command-line-arguments.TestCockroach7504
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:159 +0x27a

 Goroutine 49273 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 49273 [semacquire]:
sync.runtime_SemacquireMutex(0xc000014514, 0xc000102300, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000014510)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*tableNameCache).get(0xc000102810, 0x0)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:76 +0x103
command-line-arguments.(*LeaseManager).AcquireByName(...)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:103
command-line-arguments.TestCockroach7504.func1(0xc000060a80)
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:156 +0x38
created by command-line-arguments.TestCockroach7504
    /root/gobench/gobench/goker/blocking/cockroach/7504/cockroach7504_test.go:154 +0x24e
```

## Cockroach/9935

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[cockroach#9935]|[pull request]|[patch]| Blocking | Resource Deadlock | Double Locking |

[cockroach#9935]:(cockroach9935_test.go)
[patch]:https://github.com/cockroachdb/cockroach/pull/9935/files
[pull request]:https://github.com/cockroachdb/cockroach/pull/9935
 
### Description

This is some description from previous researchers

> This bug is caused by acquiring l.mu.Lock() twice. The fix is
> to release l.mu.Lock() before acquiring l.mu.Lock for the second time.

### Backtrace

```
goroutine 21 [semacquire]:
sync.runtime_SemacquireMutex(0xc0000b407c, 0x4d65822107fcfd00, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc0000b4078)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*loggingT).exit(0xc0000b4078, 0x5733a0, 0xc00004a010)
    /root/gobench/gobench/goker/blocking/cockroach/9935/cockroach9935_test.go:29 +0x78
command-line-arguments.(*loggingT).outputLogEntry(0xc0000b4078)
    /root/gobench/gobench/goker/blocking/cockroach/9935/cockroach9935_test.go:18 +0x85
created by command-line-arguments.TestCockroach9935
    /root/gobench/gobench/goker/blocking/cockroach/9935/cockroach9935_test.go:35 +0xa2
```

## Etcd/10492

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#10492]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[etcd#10492]:(etcd10492_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/10492/files
[pull request]:https://github.com/etcd-io/etcd/pull/10492
 
### Description

line 19, 31 double locking

### Backtrace

```
goroutine 6 [semacquire]:
sync.runtime_SemacquireMutex(0xc0000782d4, 0x0, 0x1)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc0000782d0)
	/usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
	/usr/local/go/src/sync/mutex.go:81
sync.(*RWMutex).Lock(0xc0000782d0)
	/usr/local/go/src/sync/rwmutex.go:98 +0x97
command-line-arguments.(*lessor).Checkpoint(0xc0000782d0)
	/root/gobench/goker/blocking/etcd/10492/etcd10492_test.go:20 +0x3a
command-line-arguments.TestEtcd10492.func1(0x572160, 0xc000014080)
	/root/gobench/goker/blocking/etcd/10492/etcd10492_test.go:46 +0x2a
command-line-arguments.(*lessor).Renew(0xc0000782d0)
	/root/gobench/goker/blocking/etcd/10492/etcd10492_test.go:37 +0xb9
command-line-arguments.TestEtcd10492(0xc000122120)
	/root/gobench/goker/blocking/etcd/10492/etcd10492_test.go:51 +0xf7
testing.tRunner(0xc000122120, 0x550738)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b
```

## Etcd/5509

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#5509]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[etcd#5509]:(etcd5509_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/5509/files
[pull request]:https://github.com/etcd-io/etcd/pull/5509
 
### Description

Some description from developers or pervious reseachers

> r.acquire() returns holding r.client.mu.RLock() on success; 
> it was dead locking because it was returning with the rlock held on 
> a failure path and leaking it. After that any call to client.Close() 
> will block forever waiting for the wlock.

Line 42 : Missing RUnlock before return 

## Etcd/6708

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#6708]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[etcd#6708]:(etcd6708_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6708/files
[pull request]:https://github.com/etcd-io/etcd/pull/6708
 
### Description

Line 54, 49 double locking

## Etcd/6857

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#6857]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[etcd#6857]:(etcd6857_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6857/files
[pull request]:https://github.com/etcd-io/etcd/pull/6857
 
### Description

Possible intervening

```
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
```

## Etcd/6873

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#6873]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[etcd#6873]:(etcd6873_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/6873/files
[pull request]:https://github.com/etcd-io/etcd/pull/6873
 
### Description

Possible intervening

```
///
/// G1						G2					G3
/// newWatchBroadcasts()
///	wbs.update()
/// wbs.updatec <-
/// return
///							<-wbs.updatec
///							wbs.coalesce()
///												wbs.stop()
///												wbs.mu.Lock()
///												close(wbs.updatec)
///												<-wbs.donec
///							wbs.mu.Lock()
///---------------------G2,G3 deadlock-------------------------
///
```

## Etcd/7492

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#7492]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[etcd#7492]:(etcd7492_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/7492/files
[pull request]:https://github.com/etcd-io/etcd/pull/7492
 
### Description

Possible intervening

```
///
///	G1										G2
///											stk.run()
///	ts.assignSimpleTokenToUser()
///	t.simpleTokensMu.Lock()
///	t.simpleTokenKeeper.addSimpleToken()
///	tm.addSimpleTokenCh <- true
///											<-tm.addSimpleTokenCh
///	t.simpleTokensMu.Unlock()
///	ts.assignSimpleTokenToUser()
///	...										...
///	t.simpleTokensMu.Lock()
///											<-tokenTicker.C
///	tm.addSimpleTokenCh <- true
///											tm.deleteTokenFunc()
///											t.simpleTokensMu.Lock()
///------------------------------------G1,G2 deadlock---------------------------------------------
///
```

See the [real bug](../../../../goreal/blocking/etcd/7492/README.md)

### Backtrace

```
goroutine 1077 [semacquire, 9 minutes]:
sync.runtime_Semacquire(0xc0002461a8)
	/usr/local/go/src/runtime/sema.go:56 +0x42
sync.(*WaitGroup).Wait(0xc0002461a0)
	/usr/local/go/src/sync/waitgroup.go:130 +0x64
command-line-arguments.TestEtcd7492(0xc00024ca20)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:134 +0x120
testing.tRunner(0xc00024ca20, 0x554708)
	/usr/local/go/src/testing/testing.go:1050 +0xdc
created by testing.(*T).Run
	/usr/local/go/src/testing/testing.go:1095 +0x28b

goroutine 1080 [chan send, 9 minutes]:
command-line-arguments.(*simpleTokenTTLKeeper).addSimpleToken(...)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:63
command-line-arguments.(*tokenSimple).assignSimpleTokenToUser(0xc000098240)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:84 +0x55
command-line-arguments.(*tokenSimple).assign(0xc000098240)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:79 +0x2b
command-line-arguments.(*authStore).Authenticate(...)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:28
command-line-arguments.TestEtcd7492.func1(0xc0002461a0, 0xc00005e650)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:131 +0x5c
created by command-line-arguments.TestEtcd7492
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:129 +0x104

goroutine 1078 [semacquire, 9 minutes]:
sync.runtime_SemacquireMutex(0xc00009824c, 0xc000073e01, 0x1)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000098248)
	/usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
	/usr/local/go/src/sync/mutex.go:81
sync.(*RWMutex).Lock(0xc000098248)
	/usr/local/go/src/sync/rwmutex.go:98 +0x97
command-line-arguments.newDeleterFunc.func1(0x54a981, 0x1)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:89 +0x42
command-line-arguments.(*simpleTokenTTLKeeper).run(0xc000098260)
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:52 +0xa2
created by command-line-arguments.NewSimpleTokenTTLKeeper
	/root/gobench/goker/blocking/etcd/7492/etcd7492_test.go:38 +0xd8
```

## Etcd/7902

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[etcd#7902]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[etcd#7902]:(etcd7902_test.go)
[patch]:https://github.com/etcd-io/etcd/pull/7902/files
[pull request]:https://github.com/etcd-io/etcd/pull/7902
 
### Description

Some description from developers or pervious reseachers

>  At least two goroutines are needed to trigger this bug,
>  one is leader and the other is follower. Both the leader 
>  and the follower execute the code above. If the follower
>  acquires mu.Lock() firstly and enter rc.release(), it will
>  be blocked at <- rcNextc (nextc). Only the leader can execute 
>  close(nextc) to unblock the follower inside rc.release().
>  However, in order to invoke rc.release(), the leader needs
>  to acquires mu.Lock(). 
>  The fix is to remove the lock and unlock around rc.release().

Possible intervening

```
///
/// G1						G2 (leader)					G3 (follower)
/// runElectionFunc()
/// doRounds()
/// wg.Wait()
/// 						...
/// 						mu.Lock()
/// 						rc.validate()
/// 						rcNextc = nextc
/// 						mu.Unlock()					...
/// 													mu.Lock()
/// 													rc.validate()
/// 													mu.Unlock()
/// 													mu.Lock()
/// 													rc.release()
/// 													<-rcNextc
/// 						mu.Lock()
/// -------------------------G1,G2,G3 deadlock--------------------------
///
```

### Backtrace

```
goroutine 19 [semacquire]:
sync.runtime_Semacquire(0xc000014088)
    /usr/local/go/src/runtime/sema.go:56 +0x42
sync.(*WaitGroup).Wait(0xc000014080)
    /usr/local/go/src/sync/waitgroup.go:130 +0x64
command-line-arguments.doRounds(0xc00006a060, 0x3, 0x3, 0x64)
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:59 +0xd8
command-line-arguments.runElectionFunc()
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:37 +0x326
created by command-line-arguments.TestEtcd7902
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:64 +0x88

 Goroutine 5 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 5 [semacquire]:
sync.runtime_SemacquireMutex(0xc00001407c, 0x551c00, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000014078)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.doRounds.func1(0xc000014080, 0x64, 0xc000014078, 0xc00006a060)
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:53 +0x126
created by command-line-arguments.doRounds
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:44 +0xb8

 Goroutine 6 in state chan receive, with command-line-arguments.runElectionFunc.func4 on top of the stack:
goroutine 6 [chan receive]:
command-line-arguments.runElectionFunc.func4()
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:34 +0x48
command-line-arguments.doRounds.func1(0xc000014080, 0x64, 0xc000014078, 0xc00006a080)
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:54 +0xed
created by command-line-arguments.doRounds
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:44 +0xb8

 Goroutine 7 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 7 [semacquire]:
sync.runtime_SemacquireMutex(0xc00001407c, 0x551c00, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000014078)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.doRounds.func1(0xc000014080, 0x64, 0xc000014078, 0xc00006a0a0)
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:53 +0x126
created by command-line-arguments.doRounds
    /root/gobench/goker/blocking/etcd/7902/etcd7902_test.go:44 +0xb8
```

## Grpc/1275

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#1275]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[grpc#1275]:(grpc1275_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1275/files
[pull request]:https://github.com/grpc/grpc-go/pull/1275
 
### Description

Some description from developers or pervious reseachers

> Two goroutines are invovled in this deadlock. The first goroutine
  is the main goroutine. It is blocked at case <- donec, and it is
   waiting for the second goroutine to close the channel.
   The second goroutine is created by the main goroutine. It is blocked
   when calling stream.Read(). stream.Read() invokes recvBufferRead.Read().
   The second goroutine is blocked at case i := r.recv.get(), and it is
   waiting for someone to send a message to this channel.
   It is the client.CloseSream() method called by the main goroutine that
   should send the message, but it is not. The patch is to send out this message.

Possible intervening

```
///
/// G1 									G2
/// testInflightStreamClosing()
/// 									stream.Read()
/// 									io.ReadFull()
/// 									<- r.recv.get()
/// CloseStream()
/// <- donec
/// ------------G1 timeout, G2 leak---------------------
///
```

## Grpc/1424

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#1424]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[grpc#1424]:(grpc1424_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1424/files
[pull request]:https://github.com/grpc/grpc-go/pull/1424
 
### Description

Some description from developers or pervious reseachers

> The parent function could return without draining the done channel.

Possible intervening

```
///
/// G1                      G2                          G3
/// DialContext()
///                         cc.dopts.balancer.Notify()
///                                                     cc.lbWatcher()
///                         <-doneChan
/// close()
/// -----------------------G2 leak------------------------------------
///
```

## Grpc/1460

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#1460]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[grpc#1460]:(grpc1460_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/1460/files
[pull request]:https://github.com/grpc/grpc-go/pull/1460
 
### Description

Some description from developers or pervious reseachers

> When gRPC keepalives are enabled (which isn't the case
  by default at this time) and PermitWithoutStream is false
  (the default), the client can deadlock when transitioning
  between having no active stream and having one active
  stream.The keepalive() goroutine is stuck at “<-t.awakenKeepalive”,
  while the main goroutine is stuck in NewStream() on t.mu.Lock().

Possible intervening

```
///
/// G1 						G2
/// client.keepalive()
/// 						client.NewStream()
/// t.mu.Lock()
/// <-t.awakenKeepalive
/// 						t.mu.Lock()
/// ---------------G1, G2 deadlock--------------
///
```

## Grpc/3017

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#3017]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[grpc#3017]:(grpc3017_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/3017/files
[pull request]:https://github.com/grpc/grpc-go/pull/3017
 
### Description

Line 65 missing unlock

## Grpc/660

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#660]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[grpc#660]:(grpc660_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/660/files
[pull request]:https://github.com/grpc/grpc-go/pull/660
 
### Description

Some description from developers or pervious reseachers

> The parent function could return without draining the done channel.

Possible intervening

```
///
/// G1 						G2 				helper goroutine
/// doCloseLoopUnary()
///											bc.stop <- true
/// <-bc.stop
/// return
/// 						done <-
/// ----------------------G2 leak--------------------------
///
```

## Grpc/795

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#795]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[grpc#795]:(grpc795_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/795/files
[pull request]:https://github.com/grpc/grpc-go/pull/795
 
### Description

line 20 missing unlock

## Grpc/862

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[grpc#862]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[grpc#862]:(grpc862_test.go)
[patch]:https://github.com/grpc/grpc-go/pull/862/files
[pull request]:https://github.com/grpc/grpc-go/pull/862
 
### Description

Some description from developers or pervious reseachers

> When return value conn is nil, cc (ClientConn) is not closed.
  The goroutine executing resetAddrConn is leaked. The patch is to
  close ClientConn in the defer func().

Possible intervening

```
///
/// G1 					G2
/// DialContext()
/// 					cc.resetAddrConn()
/// 					resetTransport()
/// 					<-ac.ctx.Done()
/// --------------G2 leak------------------
///
```

## Hugo/3251

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[hugo#3251]|[pull request]|[patch]| Blocking | Resource Deadlock | RWR deadlock |

[hugo#3251]:(hugo3251_test.go)
[patch]:https://github.com/gohugoio/hugo/pull/3251/files
[pull request]:https://github.com/gohugoio/hugo/pull/3251
 
### Description

A goroutine can hold Lock() at line 20 then acquire RLock() at 
line 29. RLock() at line 29 will never be acquired because Lock() 
at line 20 will never be released. 

### Backtrace

```
goroutine 9 [semacquire]:
sync.runtime_SemacquireMutex(0xc000096004, 0xc000182000, 0x1)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc000096000)
	/usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
	/usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*remoteLock).URLLock(0x645cc0, 0x54c311, 0x1a)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:25 +0xda
command-line-arguments.resGetRemote(0x54c311, 0x1a, 0x0, 0x0)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:38 +0x5f
command-line-arguments.TestHugo3251.func1(0xc000014100, 0x54c311, 0x1a, 0x2)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:54 +0x8f
created by command-line-arguments.TestHugo3251
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:51 +0xdd

goroutine 16 [semacquire]:
sync.runtime_SemacquireMutex(0x645cc4, 0x539600, 0x1)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0x645cc0)
	/usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
	/usr/local/go/src/sync/mutex.go:81
sync.(*RWMutex).Lock(0x645cc0)
	/usr/local/go/src/sync/rwmutex.go:98 +0x97
command-line-arguments.(*remoteLock).URLLock(0x645cc0, 0x54c311, 0x1a)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:21 +0x31
command-line-arguments.resGetRemote(0x54c311, 0x1a, 0x0, 0x0)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:38 +0x5f
command-line-arguments.TestHugo3251.func1(0xc000014100, 0x54c311, 0x1a, 0x9)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:54 +0x8f
created by command-line-arguments.TestHugo3251
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:51 +0xdd

goroutine 17 [semacquire]:
sync.runtime_SemacquireMutex(0x645ccc, 0xc000098000, 0x0)
	/usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*RWMutex).RLock(...)
	/usr/local/go/src/sync/rwmutex.go:50
command-line-arguments.(*remoteLock).URLUnlock(0x645cc0, 0x54c311, 0x1a)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:30 +0xf2
command-line-arguments.resGetRemote.func1(0x54c311, 0x1a)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:39 +0x41
command-line-arguments.resGetRemote(0x54c311, 0x1a, 0x0, 0x0)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:41 +0xa9
command-line-arguments.TestHugo3251.func1(0xc000014100, 0x54c311, 0x1a, 0xa)
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:54 +0x8f
created by command-line-arguments.TestHugo3251
	/root/gobench/goker/blocking/hugo/3251/hugo3251_test.go:51 +0xdd
```

## Hugo/5379

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[hugo#5379]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[hugo#5379]:(hugo5379_test.go)
[patch]:https://github.com/gohugoio/hugo/pull/5379/files
[pull request]:https://github.com/gohugoio/hugo/pull/5379
 
### Description

A goroutine first acquire `contentInitMu` at line 99 then
acquire the same Mutex at line 66

## Istio/16224

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[istio#16224]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[istio#16224]:(istio16224_test.go)
[patch]:https://github.com/istio/istio/pull/16224/files
[pull request]:https://github.com/istio/istio/pull/16224
 
### Description

A goroutine holds a mutex at line 91 and then blocked at line 93.
Another goroutine attempt acquire the same mutex at line 101 to 
further drains the same channel at 103.

## Istio/17860

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[istio#17860]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[istio#17860]:(istio17860_test.go)
[patch]:https://github.com/istio/istio/pull/17860/files
[pull request]:https://github.com/istio/istio/pull/17860
 
### Description

`a.statusCh` can't be drained at line 70.

### Backtace

```
goroutine 33 [chan send]:
command-line-arguments.(*agent).runWait(0xc000078300, 0x2)
    /root/gobench/goker/blocking/istio/17860/istio17860_test.go:71 +0x43
created by command-line-arguments.(*agent).Restart
    /root/gobench/goker/blocking/istio/17860/istio17860_test.go:67 +0xd0

 Goroutine 7 in state chan send, with command-line-arguments.(*agent).runWait on top of the stack:
goroutine 7 [chan send]:
command-line-arguments.(*agent).runWait(0xc000078300, 0x1)
    /root/gobench/goker/blocking/istio/17860/istio17860_test.go:71 +0x43
created by command-line-arguments.(*agent).Restart
    /root/gobench/goker/blocking/istio/17860/istio17860_test.go:67 +0xd0
]
```

## Istio/18454

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[istio#18454]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[istio#18454]:(istio18454_test.go)
[patch]:https://github.com/istio/istio/pull/18454/files
[pull request]:https://github.com/istio/istio/pull/18454
 
### Description

s.timer.Stop() at line 56 and 61 can be called concurrency 
(i.e. from their entry point at line 104 and line 66).
See [Timer](https://golang.org/pkg/time/#Timer).

### Backtrace

```
goroutine 5 [chan receive]:
command-line-arguments.(*Strategy).startTimer.func1(0x578320, 0xc000090800)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:58 +0x17b
command-line-arguments.(*Worker).Start.func1(0xc00005e190, 0xc0000a4240)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:22 +0x3c
created by command-line-arguments.(*Worker).Start
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:21 +0x53

 Goroutine 31 in state chan send, with command-line-arguments.(*Strategy).OnChange on top of the stack:
goroutine 31 [chan send]:
command-line-arguments.(*Strategy).OnChange(0xc0000970b0)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:43 +0x74
command-line-arguments.(*Processor).processEvent(...)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:83
command-line-arguments.(*Processor).Start.func2(0x578320, 0xc000090840)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:101 +0x78
command-line-arguments.(*Worker).Start.func1(0xc000092530, 0xc0000a4260)
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:22 +0x3c
created by command-line-arguments.(*Worker).Start
    /root/gobench/goker/blocking/istio/18454/istio18454_test.go:21 +0x53
```

## Kubernetes/10182

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#10182]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[kubernetes#10182]:(kubernetes10182_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/10182/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/10182
 
### Description

Some description from developers or pervious reseachers

>  This is a lock-channel bug. goroutine 1 is blocked on a lock
   held by goroutine 3, while goroutine 3 is blocked on sending
   message to ch, which is read by goroutine 1.

Possible intervening

```
/// G1 						G2							G3
/// s.Start()
/// s.syncBatch()
/// 						s.SetPodStatus()
/// <-s.podStatusChannel
/// 						s.podStatusesLock.Lock()
/// 						s.podStatusChannel <- true
/// 						s.podStatusesLock.Unlock()
/// 						return
/// s.DeletePodStatus()
/// 													s.podStatusesLock.Lock()
/// 													s.podStatusChannel <- true
/// s.podStatusesLock.Lock()
/// -----------------------------G1,G3 deadlock----------------------------
```

## Kubernetes/11298

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#11298]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Condition Variable |

[kubernetes#11298]:(kubernetes11298_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/11298/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/11298
 
### Description

Some description from developers or pervious reseachers

> n.node used the n.lock as underlaying locker. The service loop initially
  locked it, the Notify function tried to lock it before calling n.node.Signal,
  leading to a dead-lock.

`n.cond.Signal()` at line 59 and line 81 are not guaranteed to `n.cond.Wait` at line 56.

## Kubernetes/13135

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#13135]|[pull request]|[patch]| Blocking | Resource Deadlock | AB-BA deadlock |

[kubernetes#13135]:(kubernetes13135_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/13135/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/13135
 
### Description

```
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
```

## Kubernetes/1321

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#1321]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[kubernetes#1321]:(kubernetes1321_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/1321/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/1321
 
### Description

Some description from developers or pervious reseachers

> This is a lock-channel bug. The first goroutine invokes
> distribute() function. distribute() function holds m.lock.Lock(),
  while blocking at sending message to w.result. The second goroutine
  invokes stopWatching() funciton, which can unblock the first
  goroutine by closing w.result. However, in order to close w.result,
  stopWatching() function needs to acquire m.lock.Lock() firstly.
>
> The fix is to introduce another channel and put receive message
  from the second channel in the same select as the w.result. Close
  the second channel can unblock the first goroutine, while no need
  to hold m.lock.Lock().

Possible intervening

```
///
/// G1 							G2
/// testMuxWatcherClose()
/// NewMux()
/// 							m.loop()
/// 							m.distribute()
/// 							m.lock.Lock()
/// 							w.result <- true
/// w := m.Watch()
/// w.Stop()
/// mw.m.stopWatching()
/// m.lock.Lock()
/// ---------------G1,G2 deadlock---------------
///
```

## Kubernetes/25331

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#25331]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[kubernetes#25331]:(kubernetes25331_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/25331/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/25331
 
### Description

Some description from developers or pervious reseachers

> In reflector.go, it could probably call Stop() without retrieving
  all results from ResultChan(). See here. A potential leak is that
  when an error has happened, it could block on resultChan, and then
  cancelling context in Stop() wouldn't unblock it.

Possible intervening

```
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

```

## Kubernetes/26980

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#26980]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[kubernetes#26980]:(kubernetes26980_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/26980/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/26980
 
### Description

A goroutine holds a mutex at line 24 and blocked at line 35.
Another goroutine blocked at line 58 by acquiring the same mutex. 

## Kubernetes/30872

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#30872]|[pull request]|[patch]| Blocking | Resource Deadlock | AB-BA deadlock |

[kubernetes#30872]:(kubernetes30872_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/30872/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/30872
 
### Description

This is a AB-BA deadlock. (Lock acquires at line 92 and at line 157 respectively)

### Backtrace

```
goroutine 7 [semacquire]:
sync.runtime_SemacquireMutex(0xc00000c0a4, 0xc000072200, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc00000c0a0)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*federatedInformerImpl).addCluster(0xc00000c0a0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:93 +0x78
command-line-arguments.NewFederatedInformer.func1()
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:173 +0x2a
command-line-arguments.ResourceEventHandlerFuncs.OnAdd(0xc00005e490)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:71 +0x33
command-line-arguments.NewInformer.func1()
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:184 +0x2f
command-line-arguments.(*DeltaFIFO).Pop(0xc0000180a0, 0xc00000c0c0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:165 +0x5f
command-line-arguments.(*Controller).processLoop(...)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:53
command-line-arguments.JitterUntil.func1(...)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:25
command-line-arguments.JitterUntil(0xc000100fb0, 0xc000076300)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:26 +0x2a
command-line-arguments.Util(...)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:14
command-line-arguments.(*Controller).Run(0xc00000c0e0, 0xc000076300)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:45 +0x53
created by command-line-arguments.(*federatedInformerImpl).Start
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:102 +0xb8

 Goroutine 8 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 8 [semacquire]:
sync.runtime_SemacquireMutex(0xc00000c0a4, 0x0, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc00000c0a0)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
command-line-arguments.(*federatedInformerImpl).Stop(0xc00000c0a0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:106 +0x8e
command-line-arguments.(*NamespaceController).Run.func1(0xc0000762a0, 0xc00000c080)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:146 +0x4b
created by command-line-arguments.(*NamespaceController).Run
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:144 +0x64

 Goroutine 9 in state semacquire, with sync.runtime_SemacquireMutex on top of the stack:
goroutine 9 [semacquire]:
sync.runtime_SemacquireMutex(0xc0000180a4, 0x1, 0x1)
    /usr/local/go/src/runtime/sema.go:71 +0x47
sync.(*Mutex).lockSlow(0xc0000180a0)
    /usr/local/go/src/sync/mutex.go:138 +0xfc
sync.(*Mutex).Lock(...)
    /usr/local/go/src/sync/mutex.go:81
sync.(*RWMutex).Lock(0xc0000180a0)
    /usr/local/go/src/sync/rwmutex.go:98 +0x97
command-line-arguments.(*DeltaFIFO).HasSynced(0xc0000180a0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:158 +0x3a
command-line-arguments.(*Controller).HasSynced(0xc00000c0e0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:49 +0x33
command-line-arguments.(*federatedInformerImpl).ClustersSynced(0xc00000c0a0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:89 +0x6d
command-line-arguments.(*NamespaceController).isSynced(...)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:135
command-line-arguments.(*NamespaceController).reconcileNamespace(...)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:139
command-line-arguments.(*NamespaceController).Run.func2()
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:149 +0x34
command-line-arguments.(*DelayingDeliverer).StartWithHandler.func1(0xc00005e4a0)
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:115 +0x25
created by command-line-arguments.(*DelayingDeliverer).StartWithHandler
    /root/gobench/goker/blocking/kubernetes/30872/kubernetes30872_test.go:114 +0x3f
```

## Kubernetes/38669

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#38669]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[kubernetes#38669]:(kubernetes38669_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/38669/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/38669
 
### Description

No sender for line 33.

## Kubernetes/5316

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#5316]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[kubernetes#5316]:(kubernetes5316_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/5316/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/5316
 
### Description

Some description from developers or pervious reseachers

> If the main goroutine selects a case that doesn’t consumes
  the channels, the anonymous goroutine will be blocked on sending
  to channel.

Possible intervening

```
///
/// G1 						G2
/// finishRequest()
/// 						fn()
/// time.After()
/// 						errCh<-/ch<-
/// --------------G2 leak----------------
///
```

## Kubernetes/58107

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#58107]|[pull request]|[patch]| Blocking | Resource Deadlock | RWR deadlock |

[kubernetes#58107]:(kubernetes58107_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/58107/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/58107
 
### Description

Some description from developers or pervious reseachers

> The rules for read and write lock: allows concurrent read lock;
  write lock has higher priority than read lock.
  
> There are two queues (queue 1 and queue 2) involved in this bug,
  and the two queues are protected by the same read-write lock
  (rq.workerLock.RLock()). Before getting an element from queue 1 or
  queue 2, rq.workerLock.RLock() is acquired. If the queue is empty,
  cond.Wait() will be invoked. There is another goroutine (goroutine D),
  which will periodically invoke rq.workerLock.Lock(). Under the following
  situation, deadlock will happen. Queue 1 is empty, so that some goroutines
  hold rq.workerLock.RLock(), and block at cond.Wait(). Goroutine D is
  blocked when acquiring rq.workerLock.Lock(). Some goroutines try to process
  jobs in queue 2, but they are blocked when acquiring rq.workerLock.RLock(),
  since write lock has a higher priority.

> The fix is to not acquire rq.workerLock.RLock(), while pulling data
  from any queue. Therefore, when a goroutine is blocked at cond.Wait(),
  rq.workLock.RLock() is not held.

Possible intervening

```
/// G1 						G2						G3
/// ...						...						Sync()
/// rq.workerLock.RLock()
/// q.cond.Wait()
/// 												rq.workerLock.Lock()
/// 						rq.workerLock.RLock()
///
```

## Kubernetes/62464

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#62464]|[pull request]|[patch]| Blocking | Resource Deadlock | RWR deadlock |

[kubernetes#62464]:(kubernetes62464_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/62464/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/62464
 
### Description

Some description from developers or pervious reseachers

> This is another example for recursive read lock bug. It has
  been noticed by the go developers that RLock should not be
  recursively used in the same thread.

Possible intervening

```
///
/// G1 									G2
/// m.reconcileState()
/// m.state.GetCPUSetOrDefault()
/// s.RLock()
/// s.GetCPUSet()
/// 									p.RemoveContainer()
/// 									s.GetDefaultCPUSet()
/// 									s.SetDefaultCPUSet()
/// 									s.Lock()
/// s.RLock()
/// ---------------------G1,G2 deadlock---------------------
///
```

## Kubernetes/6632

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#6632]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[kubernetes#6632]:(kubernetes6632_test.go)
[patch]:https://github.com/kubernetes/kubernetes/pull/6632/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/6632
 
### Description

Some description from developers or pervious reseachers

> This is a lock-channel bug. When resetChan is full, WriteFrame
  holds the lock and blocks on the channel. Then monitor() fails
  to close the resetChan because lock is already held by WriteFrame.
  
> Fix: create a goroutine to drain the channel

Possible intervening

```
///
/// G1						G2					helper goroutine
/// i.monitor()
/// <-i.conn.closeChan
///							i.WriteFrame()
///							i.writeLock.Lock()
///							i.resetChan <-
///												i.conn.closeChan<-
///	i.writeLock.Lock()
///	----------------------G1,G2 deadlock------------------------
///
```

## Kubernetes/70277

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[kubernetes#70277]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[kubernetes#70277]:kubernetes70277_test.go
[patch]:https://github.com/kubernetes/kubernetes/pull/70277/files
[pull request]:https://github.com/kubernetes/kubernetes/pull/70277
 
### Description

Some description from developers or pervious reseachers

> wait.poller() returns a function with type WaitFunc. 
> the function creates a goroutine and the goroutine only 
> quits when after or done closed.

The doneCh defined at line 70 is never closed.

## Moby/17176

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#17176]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[moby#17176]:(moby17176_test.go)
[patch]:https://github.com/moby/moby/pull/17176/files
[pull request]:https://github.com/moby/moby/pull/17176
 
### Description

Some description from developers or pervious reseachers

> devices.nrDeletedDevices takes devices.Lock() but does
  not drop it if there are no deleted devices. This will block
  other goroutines trying to acquire devices.Lock().
>
> In general reason is that when device deletion is happning,
  we can try deletion/deactivation in a loop. And that that time
  we don't want to block rest of the device operations in parallel.
  So we drop the inner devices lock while continue to hold per
  device lock

Line 36 missing unlock.

## Moby/21233

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#21233]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[moby#21233]:(moby21233_test.go)
[patch]:https://github.com/moby/moby/pull/21233/files
[pull request]:https://github.com/moby/moby/pull/21233
 
### Description

Some description from developers or pervious reseachers

> This test was checking that it received every progress update that was
  produced. But delivery of these intermediate progress updates is not
  guaranteed. A new update can overwrite the previous one if the previous
  one hasn't been sent to the channel yet.
  
> The call to t.Fatalf exited the cur rent goroutine which was consuming
  the channel, which caused a deadlock and eventual test timeout rather
  than a proper failure message.

Possible intervening

```
///
/// G1 						G2					G3
/// testTransfer()
/// tm.Transfer()
/// t.Watch()
/// 						WriteProgress()
/// 						ProgressChan<-
/// 											<-progressChan
/// 						...					...
/// 						return
/// 											<-progressChan
/// <-watcher.running
/// ----------------------G1, G3 leak--------------------------
///
```

## Moby/25384

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#25384]|[pull request]|[patch]| Blocking | Mixed Deadlock | Misuse WaitGroup |

[moby#25384]:(moby25384_test.go)
[patch]:https://github.com/moby/moby/pull/25384/files
[pull request]:https://github.com/moby/moby/pull/25384
 
### Description

Some description from developers or pervious reseachers

> When n=1 (len(pm.plugins)), the location of group.Wait() doesn’t matter.
  When n is larger than 1, group.Wait() is invoked in each iteration. Whenever
  group.Wait() is invoked, it waits for group.Done() to be executed n times.
  However, group.Done() is only executed once in one iteration.

Misuse of sync.WaitGroup

## Moby/27782

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#27782]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Condition Variable |

[moby#27782]:(moby27782_test.go)
[patch]:https://github.com/moby/moby/pull/27782/files
[pull request]:https://github.com/moby/moby/pull/27782
 
### Description

Possible intervening

```
///
/// G1 						G2							G3
/// InitializeStdio()
/// startLogging()
/// l.ReadLogs()
/// NewLogWatcher()
/// 						l.readLogs()
/// container.Reset()
/// LogDriver.Close()
/// r.Close()
/// close(w.closeNotifier)
/// 						followLogs(logWatcher)
/// 						watchFile()
/// 						New()
/// 						NewEventWatcher()
/// 						NewWatcher()
/// 													w.readEvents()
/// 													event.ignoreLinux()
/// 													return false
/// 						<-logWatcher.WatchClose()
/// 						fileWatcher.Remove()
/// 						w.cv.Wait()
/// 													w.Events <- event
/// ------------------------------G2,G3 deadlock---------------------------
///

```

## Moby/28462

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#28462]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[moby#28462]:(moby28462_test.go)
[patch]:https://github.com/moby/moby/pull/28462/files
[pull request]:https://github.com/moby/moby/pull/28462
 
### Description

Some description from developers or pervious reseachers

> There are three goroutines mentioned in the bug report Moby#28405.
  Actually, only two goroutines are needed to trigger this bug. This bug
  is another example where lock and channel are mixed with each other.

Possible intervening

```
///
/// G1							G2
/// monitor()
/// handleProbeResult()
/// 							d.StateChanged()
/// 							c.Lock()
/// 							d.updateHealthMonitorElseBranch()
/// 							h.CloseMonitorChannel()
/// 							s.stop <- struct{}{}
/// c.Lock()
/// ----------------------G1,G2 deadlock------------------------
///
```

## Moby/29733

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#29733]|[pull request]|[patch]| Blocking | Communication Deadlock | Condition Variable |

[moby#29733]:(moby29733_test.go)
[patch]:https://github.com/moby/moby/pull/29733/files
[pull request]:https://github.com/moby/moby/pull/29733
 
### Description

`Wait()` at line 21 has no corresponding `Signal()`.

## Moby/30408

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#30408]|[pull request]|[patch]| Blocking | Communication Deadlock | Condition Variable |

[moby#30408]:(moby30408_test.go)
[patch]:https://github.com/moby/moby/pull/30408/files
[pull request]:https://github.com/moby/moby/pull/30408
 
### Description

`Wait()` at line 22 has no corresponding `Signal()`.

## Moby/33781

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#33781]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel & Context |

[moby#33781]:(moby33781_test.go)
[patch]:https://github.com/moby/moby/pull/33781/files
[pull request]:https://github.com/moby/moby/pull/33781
 
### Description

Some description from developers or pervious reseachers

> The goroutine created using anonymous function is blocked at
  sending message to a unbuffered channel. However there exists a
  path in the parent goroutine where the parent function will
  return without draining the channel.

Possible intervening

```
///
/// G1 				G2				G3
/// monitor()
/// <-time.After()
/// 				stop <-
/// <-stop
/// 				return
/// cancelProbe()
/// return
/// 								result<-
///----------------G3 leak------------------
///
```

## Moby/36114

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#36114]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[moby#36114]:(moby36114_test.go)
[patch]:https://github.com/moby/moby/pull/36114/files
[pull request]:https://github.com/moby/moby/pull/36114
 
### Description

Some description from developers or pervious reseachers

> This is a double lock bug. The the lock for the
  struct svm has already been locked when calling
  svm.hotRemoveVHDsAtStart()

## Moby/4951

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#4951]|[pull request]|[patch]| Blocking | Resource Deadlock | AB-BA deadlock |

[moby#4951]:(moby4951_test.go)
[patch]:https://github.com/moby/moby/pull/4951/files
[pull request]:https://github.com/moby/moby/pull/4951
 
### Description

Some description from developers or pervious reseachers

> The root cause and patch is clearly explained in the commit
  description. The global lock is devices.Lock(), and the device
  lock is baseInfo.lock.Lock(). It is very likely that this bug
  can be reproduced.

## Moby/7559

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[moby#7559]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[moby#7559]:(moby7559_test.go)
[patch]:https://github.com/moby/moby/pull/7559/files
[pull request]:https://github.com/moby/moby/pull/7559
 
### Description

Line 25 missing unlock

## Serving/2137

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[serving#2137]|[pull request]|[patch]| Blocking | Mixed Deadlock | Channel & Lock |

[serving#2137]:(serving2137_test.go)
[patch]:https://github.com/ knative/serving/pull/2137/files
[pull request]:https://github.com/ knative/serving/pull/2137
 
### Description

Possible intervening

```
//
// G1                           G2                      G3
// b.concurrentRequests(2)
// b.concurrentRequest()
// r.lock.Lock()
//                                                      start.Done()
// start.Wait()
// b.concurrentRequest()
// r.lock.Lock()
//                              start.Done()
// start.Wait()
// unlockAll(locks)
// unlock(lc)
// req.lock.Unlock()
// ok := <-req.accepted
//                              b.Maybe()
//                              b.activeRequests <- t
//                              thunk()
//                              r.lock.Lock()
//                                                      b.Maybe()
//                                                      b.activeRequests <- t
// ----------------------------G1,G2,G3 deadlock-----------------------------
//
```

## Syncthing/4829

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[syncthing#4829]|[pull request]|[patch]| Blocking | Resource Deadlock | Double locking |

[syncthing#4829]:(syncthing4829_test.go)
[patch]:https://github.com/syncthing/syncthing/pull/4829/files
[pull request]:https://github.com/syncthing/syncthing/pull/4829
 
### Description

Double locking at line 17 and line 30

## Syncthing/5795

| Bug ID|  Ref | Patch | Type | SubType | SubsubType |
| ----  | ---- | ----  | ---- | ---- | ---- |
|[syncthing#5795]|[pull request]|[patch]| Blocking | Communication Deadlock | Channel |

[syncthing#5795]:(syncthing5795_test.go)
[patch]:https://github.com/syncthing/syncthing/pull/5795/files
[pull request]:https://github.com/syncthing/syncthing/pull/5795
 
### Description

`<-c.dispatcherLoopStopped` at line 82 is blocking forever because
dispatcherLoop() is blocking at line 72.

