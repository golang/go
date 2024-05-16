// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"time"

	"golang.org/x/sync/errgroup"
	"golang.org/x/telemetry/counter"
	"golang.org/x/telemetry/internal/crashmonitor"
	"golang.org/x/telemetry/internal/telemetry"
	"golang.org/x/telemetry/internal/upload"
)

// Config controls the behavior of [Start].
type Config struct {
	// ReportCrashes, if set, will enable crash reporting.
	// ReportCrashes uses the [debug.SetCrashOutput] mechanism, which is a
	// process-wide resource.
	// Do not make other calls to that function within your application.
	// ReportCrashes is a non-functional unless the program is built with go1.23+.
	ReportCrashes bool

	// Upload causes this program to periodically upload approved counters
	// from the local telemetry database to telemetry.go.dev.
	//
	// This option has no effect unless the user has given consent
	// to enable data collection, for example by running
	// cmd/gotelemetry or affirming the gopls dialog.
	//
	// (This feature is expected to be used only by gopls.
	// Longer term, the go command may become the sole program
	// responsible for uploading.)
	Upload bool

	// TelemetryDir, if set, will specify an alternate telemetry
	// directory to write data to. If not set, it uses the default
	// directory.
	// This field is intended to be used for isolating testing environments.
	TelemetryDir string

	// UploadStartTime, if set, overrides the time used as the upload start time,
	// which is the time used by the upload logic to determine whether counter
	// file data should be uploaded. Only counter files that have expired before
	// the start time are considered for upload.
	//
	// This field can be used to simulate a future upload that collects recently
	// modified counters.
	UploadStartTime time.Time

	// UploadURL, if set, overrides the URL used to receive uploaded reports. If
	// unset, this URL defaults to https://telemetry.go.dev/upload.
	UploadURL string
}

// Start initializes telemetry using the specified configuration.
//
// Start opens the local telemetry database so that counter increment
// operations are durably recorded in the local file system.
//
// If [Config.Upload] is set, and the user has opted in to telemetry
// uploading, this process may attempt to upload approved counters
// to telemetry.go.dev.
//
// If [Config.ReportCrashes] is set, any fatal crash will be
// recorded by incrementing a counter named for the stack of the
// first running goroutine in the traceback.
//
// If either of these flags is set, Start re-executes the current
// executable as a child process, in a special mode in which it
// acts as a telemetry sidecar for the parent process (the application).
// In that mode, the call to Start will never return, so Start must
// be called immediately within main, even before such things as
// inspecting the command line. The application should avoid expensive
// steps or external side effects in init functions, as they will
// be executed twice (parent and child).
//
// Start returns a StartResult, which may be awaited via [StartResult.Wait] to
// wait for all work done by Start to complete.
func Start(config Config) *StartResult {
	if config.TelemetryDir != "" {
		telemetry.Default = telemetry.NewDir(config.TelemetryDir)
	}
	result := new(StartResult)

	mode, _ := telemetry.Default.Mode()
	if mode == "off" {
		// Telemetry is turned off. Crash reporting doesn't work without telemetry
		// at least set to "local", and the uploader isn't started in uploaderChild if
		// mode is "off"
		return result
	}

	counter.Open()

	if _, err := os.Stat(telemetry.Default.LocalDir()); err != nil {
		// There was a problem statting LocalDir, which is needed for both
		// crash monitoring and counter uploading. Most likely, there was an
		// error creating telemetry.LocalDir in the counter.Open call above.
		// Don't start the child.
		return result
	}

	// Crash monitoring and uploading both require a sidecar process.
	if (config.ReportCrashes && crashmonitor.Supported()) || (config.Upload && mode != "off") {
		switch v := os.Getenv(telemetryChildVar); v {
		case "":
			// The subprocess started by parent has X_TELEMETRY_CHILD=1.
			parent(config, result)
		case "1":
			// golang/go#67211: be sure to set telemetryChildVar before running the
			// child, because the child itself invokes the go command to download the
			// upload config. If the telemetryChildVar variable is still set to "1",
			// that delegated go command may think that it is itself a telemetry
			// child.
			//
			// On the other hand, if telemetryChildVar were simply unset, then the
			// delegated go commands would fork themselves recursively. Short-circuit
			// this recursion.
			os.Setenv(telemetryChildVar, "2")
			child(config)
			os.Exit(0)
		case "2":
			// Do nothing: see note above.
		default:
			log.Fatalf("unexpected value for %q: %q", telemetryChildVar, v)
		}
	}
	return result
}

// A StartResult is a handle to the result of a call to [Start]. Call
// [StartResult.Wait] to wait for the completion of all work done on behalf of
// Start.
type StartResult struct {
	wg sync.WaitGroup
}

// Wait waits for the completion of all work initiated by [Start].
func (res *StartResult) Wait() {
	if res == nil {
		return
	}
	res.wg.Wait()
}

var daemonize = func(cmd *exec.Cmd) {}

// If telemetryChildVar is set to "1" in the environment, this is the telemetry
// child.
//
// If telemetryChildVar is set to "2", this is a child of the child, and no
// further forking should occur.
const telemetryChildVar = "X_TELEMETRY_CHILD"

func parent(config Config, result *StartResult) {
	// This process is the application (parent).
	// Fork+exec the telemetry child.
	exe, err := os.Executable()
	if err != nil {
		// There was an error getting os.Executable. It's possible
		// for this to happen on AIX if os.Args[0] is not an absolute
		// path and we can't find os.Args[0] in PATH.
		log.Printf("failed to start telemetry sidecar: os.Executable: %v", err)
		return
	}
	cmd := exec.Command(exe, "** telemetry **") // this unused arg is just for ps(1)
	daemonize(cmd)
	cmd.Env = append(os.Environ(), telemetryChildVar+"=1")
	cmd.Dir = telemetry.Default.LocalDir()

	// The child process must write to a log file, not
	// the stderr file it inherited from the parent, as
	// the child may outlive the parent but should not prolong
	// the life of any pipes created (by the grandparent)
	// to gather the output of the parent.
	//
	// By default, we discard the child process's stderr,
	// but in line with the uploader, log to a file in local/debug
	// only if that directory was created by the user.
	localDebug := filepath.Join(telemetry.Default.LocalDir(), "debug")
	fd, err := os.Stat(localDebug)
	if err != nil {
		if !os.IsNotExist(err) {
			log.Fatalf("failed to stat debug directory: %v", err)
		}
	} else if fd.IsDir() {
		// local/debug exists and is a directory. Set stderr to a log file path
		// in local/debug.
		childLogPath := filepath.Join(localDebug, "sidecar.log")
		childLog, err := os.OpenFile(childLogPath, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0600)
		if err != nil {
			log.Fatalf("opening sidecar log file for child: %v", err)
		}
		defer childLog.Close()
		cmd.Stderr = childLog
	}

	if config.ReportCrashes {
		pipe, err := cmd.StdinPipe()
		if err != nil {
			log.Fatalf("StdinPipe: %v", err)
		}

		crashmonitor.Parent(pipe.(*os.File)) // (this conversion is safe)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("can't start telemetry child process: %v", err)
	}
	result.wg.Add(1)
	go func() {
		cmd.Wait() // Release resources if cmd happens not to outlive this process.
		result.wg.Done()
	}()
}

func child(config Config) {
	log.SetPrefix(fmt.Sprintf("telemetry-sidecar (pid %v): ", os.Getpid()))

	// Start crashmonitoring and uploading depending on what's requested
	// and wait for the longer running child to complete before exiting:
	// if we collected a crash before the upload finished, wait for the
	// upload to finish before exiting
	var g errgroup.Group

	if config.Upload {
		g.Go(func() error {
			uploaderChild(config.UploadStartTime, config.UploadURL)
			return nil
		})
	}
	if config.ReportCrashes {
		g.Go(func() error {
			crashmonitor.Child()
			return nil
		})
	}
	g.Wait()
}

func uploaderChild(asof time.Time, uploadURL string) {
	if mode, _ := telemetry.Default.Mode(); mode == "off" {
		// There's no work to be done if telemetry is turned off.
		return
	}
	if telemetry.Default.LocalDir() == "" {
		// The telemetry dir wasn't initialized properly, probably because
		// os.UserConfigDir did not complete successfully. In that case
		// there are no counters to upload, so we should just do nothing.
		return
	}
	tokenfilepath := filepath.Join(telemetry.Default.LocalDir(), "upload.token")
	ok, err := acquireUploadToken(tokenfilepath)
	if err != nil {
		log.Printf("error acquiring upload token: %v", err)
		return
	} else if !ok {
		// It hasn't been a day since the last upload.Run attempt or there's
		// a concurrently running uploader.
		return
	}
	if err := upload.Run(upload.RunConfig{
		UploadURL: uploadURL,
		LogWriter: os.Stderr,
		StartTime: asof,
	}); err != nil {
		log.Printf("upload failed: %v", err)
	}
}

// acquireUploadToken acquires a token permitting the caller to upload.
// To limit the frequency of uploads, only one token is issue per
// machine per time period.
// The boolean indicates whether the token was acquired.
func acquireUploadToken(tokenfile string) (bool, error) {
	const period = 24 * time.Hour

	// A process acquires a token by successfully creating a
	// well-known file. If the file already exists and has an
	// mtime age less then than the period, the process does
	// not acquire the token. If the file is older than the
	// period, the process is allowed to remove the file and
	// try to re-create it.
	fi, err := os.Stat(tokenfile)
	if err == nil {
		if time.Since(fi.ModTime()) < period {
			return false, nil
		}
		// There's a possible race here where two processes check the
		// token file and see that it's older than the period, then the
		// first one removes it and creates another, and then a second one
		// removes the newly created file and creates yet another
		// file. Then both processes would act as though they had the token.
		// This is very rare, but it's also okay because we're only grabbing
		// the token to do rate limiting, not for correctness.
		_ = os.Remove(tokenfile)
	} else if !os.IsNotExist(err) {
		return false, fmt.Errorf("statting token file: %v", err)
	}

	f, err := os.OpenFile(tokenfile, os.O_CREATE|os.O_EXCL, 0666)
	if err != nil {
		if os.IsExist(err) {
			return false, nil
		}
		return false, fmt.Errorf("creating token file: %v", err)
	}
	_ = f.Close()
	return true, nil
}
