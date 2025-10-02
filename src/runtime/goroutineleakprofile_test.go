// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
)

func TestGoroutineLeakProfile(t *testing.T) {
	// Goroutine leak test case.
	//
	// Test cases can be configured with test name, the name of the entry point function,
	// a set of expected leaks identified by regular expressions, and the number of times
	// the test should be repeated.
	//
	// Repeated runs reduce flakiness in some tests.
	type testCase struct {
		name          string
		simple        bool
		repetitions   int
		expectedLeaks map[*regexp.Regexp]bool

		// flakyLeaks are goroutine leaks that are too flaky to be reliably detected.
		// Still, they might pop up every once in a while. The test will pass regardless
		// if they occur or nor, as they are not unexpected.
		//
		// Note that all flaky leaks are true positives, i.e. real goroutine leaks,
		// and it is only their detection that is unreliable due to scheduling
		// non-determinism.
		flakyLeaks map[*regexp.Regexp]struct{}
	}

	// makeAnyTest is a short-hand for creating test cases.
	// Each of the leaks in the list is identified by a regular expression.
	// If a leak is flaky, it is added to the flakyLeaks map.
	makeAnyTest := func(name string, flaky bool, repetitions int, leaks ...string) testCase {
		tc := testCase{
			name:          name,
			expectedLeaks: make(map[*regexp.Regexp]bool, len(leaks)),
			flakyLeaks:    make(map[*regexp.Regexp]struct{}, len(leaks)),
			// Make sure the test is repeated at least once.
			repetitions: repetitions | 1,
		}

		for _, leak := range leaks {
			if !flaky {
				tc.expectedLeaks[regexp.MustCompile(leak)] = false
			} else {
				tc.flakyLeaks[regexp.MustCompile(leak)] = struct{}{}
			}
		}

		return tc
	}

	// makeTest is a short-hand for creating non-flaky test cases.
	makeTest := func(name string, leaks ...string) testCase {
		tcase := makeAnyTest(name, false, 2, leaks...)
		tcase.simple = true
		return tcase
	}

	// makeFlakyTest is a short-hand for creating flaky test cases.
	makeFlakyTest := func(name string, leaks ...string) testCase {
		if testing.Short() {
			return makeAnyTest(name, true, 2, leaks...)
		}
		return makeAnyTest(name, true, 10, leaks...)
	}

	goroutineHeader := regexp.MustCompile(`goroutine \d+ \[`)

	// extractLeaks takes the output of a test and splits it into a
	// list of strings denoting goroutine leaks.
	//
	// If the input is:
	//
	// goroutine 1 [wait reason (leaked)]:
	// main.leaked()
	// 	./testdata/testgoroutineleakprofile/foo.go:37 +0x100
	// created by main.main()
	// 	./testdata/testgoroutineleakprofile/main.go:10 +0x20
	//
	// goroutine 2 [wait reason (leaked)]:
	// main.leaked2()
	// 	./testdata/testgoroutineleakprofile/foo.go:37 +0x100
	// created by main.main()
	// 	./testdata/testgoroutineleakprofile/main.go:10 +0x20
	//
	// The output is (as a list of strings):
	//
	// leaked() [wait reason]
	// leaked2() [wait reason]
	extractLeaks := func(output string) []string {
		stacks := strings.Split(output, "\n\ngoroutine")
		var leaks []string
		for _, stack := range stacks {
			lines := strings.Split(stack, "\n")
			if len(lines) < 5 {
				// Expecting at least the following lines (where n=len(lines)-1):
				//
				// [0] goroutine n [wait reason (leaked)]
				// ...
				// [n-3] bottom.leak.frame(...)
				// [n-2]  ./bottom/leak/frame/source.go:line
				// [n-1] created by go.instruction()
				// [n] 	  ./go/instruction/source.go:line
				continue
			}

			if !strings.Contains(lines[0], "(leaked)") {
				// Ignore non-leaked goroutines.
				continue
			}

			// Get the wait reason from the goroutine header.
			header := lines[0]
			waitReason := goroutineHeader.ReplaceAllString(header, "[")
			waitReason = strings.ReplaceAll(waitReason, " (leaked)", "")

			// Get the function name from the stack trace (should be two lines above `created by`).
			var funcName string
			for i := len(lines) - 1; i >= 0; i-- {
				if strings.Contains(lines[i], "created by") {
					funcName = strings.TrimPrefix(lines[i-2], "main.")
					break
				}
			}
			if funcName == "" {
				t.Fatalf("failed to extract function name from stack trace: %s", lines)
			}

			leaks = append(leaks, funcName+" "+waitReason)
		}
		return leaks
	}

	// Micro tests involve very simple leaks for each type of concurrency primitive operation.
	microTests := []testCase{
		makeTest("NilRecv",
			`NilRecv\.func1\(.* \[chan receive \(nil chan\)\]`,
		),
		makeTest("NilSend",
			`NilSend\.func1\(.* \[chan send \(nil chan\)\]`,
		),
		makeTest("SelectNoCases",
			`SelectNoCases\.func1\(.* \[select \(no cases\)\]`,
		),
		makeTest("ChanRecv",
			`ChanRecv\.func1\(.* \[chan receive\]`,
		),
		makeTest("ChanSend",
			`ChanSend\.func1\(.* \[chan send\]`,
		),
		makeTest("Select",
			`Select\.func1\(.* \[select\]`,
		),
		makeTest("WaitGroup",
			`WaitGroup\.func1\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeTest("MutexStack",
			`MutexStack\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("MutexHeap",
			`MutexHeap\.func1.1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Cond",
			`Cond\.func1\(.* \[sync\.Cond\.Wait\]`,
		),
		makeTest("RWMutexRLock",
			`RWMutexRLock\.func1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeTest("RWMutexLock",
			`RWMutexLock\.func1\(.* \[sync\.(RW)?Mutex\.Lock\]`,
		),
		makeTest("Mixed",
			`Mixed\.func1\(.* \[sync\.WaitGroup\.Wait\]`,
			`Mixed\.func1.1\(.* \[chan send\]`,
		),
		makeTest("NoLeakGlobal"),
	}

	// Stress tests are flaky and we do not strictly care about their output.
	// They are only intended to stress the goroutine leak detector and profiling
	// infrastructure in interesting ways.
	stressTestCases := []testCase{
		makeFlakyTest("SpawnGC",
			`spawnGC.func1\(.* \[chan receive\]`,
		),
		makeTest("DaisyChain"),
	}

	// Common goroutine leak patterns.
	// Extracted from "Unveiling and Vanquishing Goroutine Leaks in Enterprise Microservices: A Dynamic Analysis Approach"
	// doi:10.1109/CGO57630.2024.10444835
	patternTestCases := []testCase{
		makeTest("NoCloseRange",
			`noCloseRange\(.* \[chan send\]`,
			`noCloseRange\.func1\(.* \[chan receive\]`,
		),
		makeTest("MethodContractViolation",
			`worker\.Start\.func1\(.* \[select\]`,
		),
		makeTest("DoubleSend",
			`DoubleSend\.func3\(.* \[chan send\]`,
		),
		makeTest("EarlyReturn",
			`earlyReturn\.func1\(.* \[chan send\]`,
		),
		makeTest("NCastLeak",
			`nCastLeak\.func1\(.* \[chan send\]`,
			`NCastLeak\.func2\(.* \[chan receive\]`,
		),
		makeTest("Timeout",
			// (vsaioc): Timeout is *theoretically* flaky, but the
			// pseudo-random choice for select case branches makes it
			// practically impossible for it to fail.
			`timeout\.func1\(.* \[chan send\]`,
		),
	}

	// GoKer tests from "GoBench: A Benchmark Suite of Real-World Go Concurrency Bugs".
	// Refer to testdata/testgoroutineleakprofile/goker/README.md.
	//
	// This list is curated for tests that are not excessively flaky.
	// Some tests are also excluded because they are redundant.
	//
	// TODO(vsaioc): Some of these might be removable (their patterns may overlap).
	gokerTestCases := []testCase{
		makeFlakyTest("Cockroach584",
			`Cockroach584\.func2\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach1055",
			`Cockroach1055\.func2\(.* \[chan receive\]`,
			`Cockroach1055\.func2\.2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Cockroach1055\.func2\.1\(.* \[chan receive\]`,
			`Cockroach1055\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach1462",
			`\(\*Stopper_cockroach1462\)\.RunWorker\.func1\(.* \[chan send\]`,
			`Cockroach1462\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeFlakyTest("Cockroach2448",
			`\(\*Store_cockroach2448\)\.processRaft\(.* \[select\]`,
			`\(\*state_cockroach2448\)\.start\(.* \[select\]`,
		),
		makeFlakyTest("Cockroach3710",
			`\(\*Store_cockroach3710\)\.ForceRaftLogScanAndProcess\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*Store_cockroach3710\)\.processRaft\.func1\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach6181",
			`testRangeCacheCoalescedRequests_cockroach6181\(.* \[sync\.WaitGroup\.Wait\]`,
			`testRangeCacheCoalescedRequests_cockroach6181\.func1\.1\(.* \[sync\.(RW)?Mutex\.Lock\]`,
			`testRangeCacheCoalescedRequests_cockroach6181\.func1\.1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeTest("Cockroach7504",
			`Cockroach7504\.func2\.1.* \[sync\.Mutex\.Lock\]`,
			`Cockroach7504\.func2\.2.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach9935",
			`\(\*loggingT_cockroach9935\)\.outputLogEntry\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach10214",
			`\(*Store_cockroach10214\)\.sendQueuedHeartbeats\(.* \[sync\.Mutex\.Lock\]`,
			`\(*Replica_cockroach10214\)\.tick\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach10790",
			`\(\*Replica_cockroach10790\)\.beginCmds\.func1\(.* \[chan receive\]`,
		),
		makeTest("Cockroach13197",
			`\(\*Tx_cockroach13197\)\.awaitDone\(.* \[chan receive\]`,
		),
		makeTest("Cockroach13755",
			`\(\*Rows_cockroach13755\)\.awaitDone\(.* \[chan receive\]`,
		),
		makeFlakyTest("Cockroach16167",
			`Cockroach16167\.func2\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*Executor_cockroach16167\)\.Start\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Cockroach18101",
			`restore_cockroach18101\.func1\(.* \[chan send\]`,
		),
		makeTest("Cockroach24808",
			`Cockroach24808\.func2\(.* \[chan send\]`,
		),
		makeTest("Cockroach25456",
			`Cockroach25456\.func2\(.* \[chan receive\]`,
		),
		makeTest("Cockroach35073",
			`Cockroach35073\.func2.1\(.* \[chan send\]`,
			`Cockroach35073\.func2\(.* \[chan send\]`,
		),
		makeTest("Cockroach35931",
			`Cockroach35931\.func2\(.* \[chan send\]`,
		),
		makeTest("Etcd5509",
			`Etcd5509\.func2\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeTest("Etcd6708",
			`Etcd6708\.func2\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeFlakyTest("Etcd6857",
			`\(\*node_etcd6857\)\.Status\(.* \[chan send\]`,
		),
		makeFlakyTest("Etcd6873",
			`\(\*watchBroadcasts_etcd6873\)\.stop\(.* \[chan receive\]`,
			`newWatchBroadcasts_etcd6873\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Etcd7492",
			`Etcd7492\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Etcd7492\.func2\.1\(.* \[chan send\]`,
			`\(\*simpleTokenTTLKeeper_etcd7492\)\.run\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Etcd7902",
			`doRounds_etcd7902\.func1\(.* \[chan receive\]`,
			`doRounds_etcd7902\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`runElectionFunc_etcd7902\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeTest("Etcd10492",
			`Etcd10492\.func2\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Grpc660",
			`\(\*benchmarkClient_grpc660\)\.doCloseLoopUnary\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Grpc795",
			`\(\*Server_grpc795\)\.Serve\(.* \[sync\.Mutex\.Lock\]`,
			`testServerGracefulStopIdempotent_grpc795\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Grpc862",
			`DialContext_grpc862\.func2\(.* \[chan receive\]`),
		makeTest("Grpc1275",
			`testInflightStreamClosing_grpc1275\.func1\(.* \[chan receive\]`),
		makeTest("Grpc1424",
			`DialContext_grpc1424\.func1\(.* \[chan receive\]`),
		makeFlakyTest("Grpc1460",
			`\(\*http2Client_grpc1460\)\.keepalive\(.* \[chan receive\]`,
			`\(\*http2Client_grpc1460\)\.NewStream\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Grpc3017",
			// grpc/3017 involves a goroutine leak that also simultaneously engages many GC assists.
			`Grpc3017\.func2\(.* \[chan receive\]`,
			`Grpc3017\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*lbCacheClientConn_grpc3017\)\.RemoveSubConn\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Hugo3251",
			`Hugo3251\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
			`Hugo3251\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`Hugo3251\.func2\.1\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeFlakyTest("Hugo5379",
			`\(\*Page_hugo5379\)\.initContent\.func1\.1\(.* \[sync\.Mutex\.Lock\]`,
			`pageRenderer_hugo5379\(.* \[sync\.Mutex\.Lock\]`,
			`Hugo5379\.func2\(.* \[sync\.WaitGroup\.Wait\]`,
	  ),
		makeFlakyTest("Istio16224",
			`Istio16224\.func2\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*controller_istio16224\)\.Run\(.* \[chan send\]`,
			`\(\*controller_istio16224\)\.Run\(.* \[chan receive\]`,
		),
		makeFlakyTest("Istio17860",
			`\(\*agent_istio17860\)\.runWait\(.* \[chan send\]`,
		),
		makeFlakyTest("Istio18454",
			`\(\*Worker_istio18454\)\.Start\.func1\(.* \[chan receive\]`,
			`\(\*Worker_istio18454\)\.Start\.func1\(.* \[chan send\]`,
		),
		// NOTE(vsaioc):
		// Kubernetes/1321 is excluded due to a race condition in the original program
		// that may, in extremely rare cases, lead to nil pointer dereference crashes.
		// (Reproducible even with regular GC). Only kept here for posterity.
		//
		// makeTest(testCase{name: "Kubernetes1321"},
		// 	`NewMux_kubernetes1321\.gowrap1\(.* \[chan send\]`,
		// 	`testMuxWatcherClose_kubernetes1321\(.* \[sync\.Mutex\.Lock\]`),
		makeTest("Kubernetes5316",
			`finishRequest_kubernetes5316\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes6632",
			`\(\*idleAwareFramer_kubernetes6632\)\.monitor\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*idleAwareFramer_kubernetes6632\)\.WriteFrame\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes10182",
			`\(\*statusManager_kubernetes10182\)\.Start\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*statusManager_kubernetes10182\)\.SetPodStatus\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes11298",
			`After_kubernetes11298\.func1\(.* \[chan receive\]`,
			`After_kubernetes11298\.func1\(.* \[sync\.Cond\.Wait\]`,
			`Kubernetes11298\.func2\(.* \[chan receive\]`,
		),
		makeFlakyTest("Kubernetes13135",
			`Util_kubernetes13135\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*WatchCache_kubernetes13135\)\.Add\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Kubernetes25331",
			`\(\*watchChan_kubernetes25331\)\.run\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes26980",
			`Kubernetes26980\.func2\(.* \[chan receive\]`,
			`Kubernetes26980\.func2\.1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*processorListener_kubernetes26980\)\.pop\(.* \[chan receive\]`,
		),
		makeFlakyTest("Kubernetes30872",
			`\(\*DelayingDeliverer_kubernetes30872\)\.StartWithHandler\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*Controller_kubernetes30872\)\.Run\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*NamespaceController_kubernetes30872\)\.Run\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Kubernetes38669",
			`\(\*cacheWatcher_kubernetes38669\)\.process\(.* \[chan send\]`,
		),
		makeFlakyTest("Kubernetes58107",
			`\(\*ResourceQuotaController_kubernetes58107\)\.worker\(.* \[sync\.Cond\.Wait\]`,
			`\(\*ResourceQuotaController_kubernetes58107\)\.worker\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*ResourceQuotaController_kubernetes58107\)\.Sync\(.* \[sync\.RWMutex\.Lock\]`,
		),
		makeFlakyTest("Kubernetes62464",
			`\(\*manager_kubernetes62464\)\.reconcileState\(.* \[sync\.RWMutex\.RLock\]`,
			`\(\*staticPolicy_kubernetes62464\)\.RemoveContainer\(.* \[sync\.(RW)?Mutex\.Lock\]`,
		),
		makeFlakyTest("Kubernetes70277",
			`Kubernetes70277\.func2\(.* \[chan receive\]`,
		),
		makeFlakyTest("Moby4951",
			`\(\*DeviceSet_moby4951\)\.DeleteDevice\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Moby7559",
			`\(\*UDPProxy_moby7559\)\.Run\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeTest("Moby17176",
			`testDevmapperLockReleasedDeviceDeletion_moby17176\.func1\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Moby21233",
			`\(\*Transfer_moby21233\)\.Watch\.func1\(.* \[chan send\]`,
			`\(\*Transfer_moby21233\)\.Watch\.func1\(.* \[select\]`,
			`testTransfer_moby21233\(.* \[chan receive\]`,
		),
		makeTest("Moby25348",
			`\(\*Manager_moby25348\)\.init\(.* \[sync\.WaitGroup\.Wait\]`,
		),
		makeFlakyTest("Moby27782",
			`\(\*JSONFileLogger_moby27782\)\.readLogs\(.* \[sync\.Cond\.Wait\]`,
			`\(\*Watcher_moby27782\)\.readEvents\(.* \[select\]`,
		),
		makeFlakyTest("Moby28462",
			`monitor_moby28462\(.* \[sync\.Mutex\.Lock\]`,
			`\(\*Daemon_moby28462\)\.StateChanged\(.* \[chan send\]`,
		),
		makeTest("Moby30408",
			`Moby30408\.func2\(.* \[chan receive\]`,
			`testActive_moby30408\.func1\(.* \[sync\.Cond\.Wait\]`,
		),
		makeFlakyTest("Moby33781",
			`monitor_moby33781\.func1\(.* \[chan send\]`,
		),
		makeFlakyTest("Moby36114",
			`\(\*serviceVM_moby36114\)\.hotAddVHDsAtStart\(.* \[sync\.Mutex\.Lock\]`,
		),
		makeFlakyTest("Serving2137",
			`\(\*Breaker_serving2137\)\.concurrentRequest\.func1\(.* \[chan send\]`,
			`\(\*Breaker_serving2137\)\.concurrentRequest\.func1\(.* \[sync\.Mutex\.Lock\]`,
			`Serving2137\.func2\(.* \[chan receive\]`,
		),
		makeTest("Syncthing4829",
			`Syncthing4829\.func2\(.* \[sync\.RWMutex\.RLock\]`,
		),
		makeTest("Syncthing5795",
			`\(\*rawConnection_syncthing5795\)\.dispatcherLoop\(.* \[chan receive\]`,
			`Syncthing5795\.func2.* \[chan receive\]`,
		),
	}

	// Combine all test cases into a single list.
	testCases := append(microTests, stressTestCases...)
	testCases = append(testCases, patternTestCases...)

	// Test cases must not panic or cause fatal exceptions.
	failStates := regexp.MustCompile(`fatal|panic`)

	testApp := func(exepath string, testCases []testCase) {

		// Build the test program once.
		exe, err := buildTestProg(t, exepath)
		if err != nil {
			t.Fatal(fmt.Sprintf("building testgoroutineleakprofile failed: %v", err))
		}

		for _, tcase := range testCases {
			t.Run(tcase.name, func(t *testing.T) {
				t.Parallel()

				cmdEnv := []string{
					"GODEBUG=asyncpreemptoff=1",
					"GOEXPERIMENT=greenteagc,goroutineleakprofile",
				}

				if tcase.simple {
					// If the test is simple, set GOMAXPROCS=1 in order to better
					// control the behavior of the scheduler.
					cmdEnv = append(cmdEnv, "GOMAXPROCS=1")
				}

				var output string
				for i := 0; i < tcase.repetitions; i++ {
					// Run program for one repetition and get runOutput trace.
					runOutput := runBuiltTestProg(t, exe, tcase.name, cmdEnv...)
					if len(runOutput) == 0 {
						t.Errorf("Test %s produced no output. Is the goroutine leak profile collected?", tcase.name)
					}

					// Zero tolerance policy for fatal exceptions or panics.
					if failStates.MatchString(runOutput) {
						t.Errorf("unexpected fatal exception or panic!\noutput:\n%s\n\n", runOutput)
					}

					output += runOutput + "\n\n"
				}

				// Extract all the goroutine leaks
				foundLeaks := extractLeaks(output)

				// If the test case was not expected to produce leaks, but some were reported,
				// stop the test immediately. Zero tolerance policy for false positives.
				if len(tcase.expectedLeaks)+len(tcase.flakyLeaks) == 0 && len(foundLeaks) > 0 {
					t.Errorf("output:\n%s\n\ngoroutines leaks detected in case with no leaks", output)
				}

				unexpectedLeaks := make([]string, 0, len(foundLeaks))

				// Parse every leak and check if it is expected (maybe as a flaky leak).
			LEAKS:
				for _, leak := range foundLeaks {
					// Check if the leak is expected.
					// If it is, check whether it has been encountered before.
					var foundNew bool
					var leakPattern *regexp.Regexp

					for expectedLeak, ok := range tcase.expectedLeaks {
						if expectedLeak.MatchString(leak) {
							if !ok {
								foundNew = true
							}

							leakPattern = expectedLeak
							break
						}
					}

					if foundNew {
						// Only bother writing if we found a new leak.
						tcase.expectedLeaks[leakPattern] = true
					}

					if leakPattern == nil {
						// We are dealing with a leak not marked as expected.
						// Check if it is a flaky leak.
						for flakyLeak := range tcase.flakyLeaks {
							if flakyLeak.MatchString(leak) {
								// The leak is flaky. Carry on to the next line.
								continue LEAKS
							}
						}

						unexpectedLeaks = append(unexpectedLeaks, leak)
					}
				}

				missingLeakStrs := make([]string, 0, len(tcase.expectedLeaks))
				for expectedLeak, found := range tcase.expectedLeaks {
					if !found {
						missingLeakStrs = append(missingLeakStrs, expectedLeak.String())
					}
				}

				var errors []error
				if len(unexpectedLeaks) > 0 {
					errors = append(errors, fmt.Errorf("unexpected goroutine leaks:\n%s\n", strings.Join(unexpectedLeaks, "\n")))
				}
				if len(missingLeakStrs) > 0 {
					errors = append(errors, fmt.Errorf("missing expected leaks:\n%s\n", strings.Join(missingLeakStrs, ", ")))
				}
				if len(errors) > 0 {
					t.Fatalf("Failed with the following errors:\n%s\n\noutput:\n%s", errors, output)
				}
			})
		}
	}

	testApp("testgoroutineleakprofile", testCases)
	testApp("testgoroutineleakprofile/goker", gokerTestCases)
}
