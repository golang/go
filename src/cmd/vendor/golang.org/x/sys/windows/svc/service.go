// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

// Package svc provides everything required to build Windows service.
//
package svc

import (
	"errors"
	"runtime"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

// State describes service execution state (Stopped, Running and so on).
type State uint32

const (
	Stopped         = State(windows.SERVICE_STOPPED)
	StartPending    = State(windows.SERVICE_START_PENDING)
	StopPending     = State(windows.SERVICE_STOP_PENDING)
	Running         = State(windows.SERVICE_RUNNING)
	ContinuePending = State(windows.SERVICE_CONTINUE_PENDING)
	PausePending    = State(windows.SERVICE_PAUSE_PENDING)
	Paused          = State(windows.SERVICE_PAUSED)
)

// Cmd represents service state change request. It is sent to a service
// by the service manager, and should be actioned upon by the service.
type Cmd uint32

const (
	Stop                  = Cmd(windows.SERVICE_CONTROL_STOP)
	Pause                 = Cmd(windows.SERVICE_CONTROL_PAUSE)
	Continue              = Cmd(windows.SERVICE_CONTROL_CONTINUE)
	Interrogate           = Cmd(windows.SERVICE_CONTROL_INTERROGATE)
	Shutdown              = Cmd(windows.SERVICE_CONTROL_SHUTDOWN)
	ParamChange           = Cmd(windows.SERVICE_CONTROL_PARAMCHANGE)
	NetBindAdd            = Cmd(windows.SERVICE_CONTROL_NETBINDADD)
	NetBindRemove         = Cmd(windows.SERVICE_CONTROL_NETBINDREMOVE)
	NetBindEnable         = Cmd(windows.SERVICE_CONTROL_NETBINDENABLE)
	NetBindDisable        = Cmd(windows.SERVICE_CONTROL_NETBINDDISABLE)
	DeviceEvent           = Cmd(windows.SERVICE_CONTROL_DEVICEEVENT)
	HardwareProfileChange = Cmd(windows.SERVICE_CONTROL_HARDWAREPROFILECHANGE)
	PowerEvent            = Cmd(windows.SERVICE_CONTROL_POWEREVENT)
	SessionChange         = Cmd(windows.SERVICE_CONTROL_SESSIONCHANGE)
)

// Accepted is used to describe commands accepted by the service.
// Note that Interrogate is always accepted.
type Accepted uint32

const (
	AcceptStop                  = Accepted(windows.SERVICE_ACCEPT_STOP)
	AcceptShutdown              = Accepted(windows.SERVICE_ACCEPT_SHUTDOWN)
	AcceptPauseAndContinue      = Accepted(windows.SERVICE_ACCEPT_PAUSE_CONTINUE)
	AcceptParamChange           = Accepted(windows.SERVICE_ACCEPT_PARAMCHANGE)
	AcceptNetBindChange         = Accepted(windows.SERVICE_ACCEPT_NETBINDCHANGE)
	AcceptHardwareProfileChange = Accepted(windows.SERVICE_ACCEPT_HARDWAREPROFILECHANGE)
	AcceptPowerEvent            = Accepted(windows.SERVICE_ACCEPT_POWEREVENT)
	AcceptSessionChange         = Accepted(windows.SERVICE_ACCEPT_SESSIONCHANGE)
)

// Status combines State and Accepted commands to fully describe running service.
type Status struct {
	State      State
	Accepts    Accepted
	CheckPoint uint32 // used to report progress during a lengthy operation
	WaitHint   uint32 // estimated time required for a pending operation, in milliseconds
}

// ChangeRequest is sent to the service Handler to request service status change.
type ChangeRequest struct {
	Cmd           Cmd
	EventType     uint32
	EventData     uintptr
	CurrentStatus Status
}

// Handler is the interface that must be implemented to build Windows service.
type Handler interface {

	// Execute will be called by the package code at the start of
	// the service, and the service will exit once Execute completes.
	// Inside Execute you must read service change requests from r and
	// act accordingly. You must keep service control manager up to date
	// about state of your service by writing into s as required.
	// args contains service name followed by argument strings passed
	// to the service.
	// You can provide service exit code in exitCode return parameter,
	// with 0 being "no error". You can also indicate if exit code,
	// if any, is service specific or not by using svcSpecificEC
	// parameter.
	Execute(args []string, r <-chan ChangeRequest, s chan<- Status) (svcSpecificEC bool, exitCode uint32)
}

var (
	// These are used by asm code.
	goWaitsH                       uintptr
	cWaitsH                        uintptr
	ssHandle                       uintptr
	sName                          *uint16
	sArgc                          uintptr
	sArgv                          **uint16
	ctlHandlerExProc               uintptr
	cSetEvent                      uintptr
	cWaitForSingleObject           uintptr
	cRegisterServiceCtrlHandlerExW uintptr
)

func init() {
	k := syscall.MustLoadDLL("kernel32.dll")
	cSetEvent = k.MustFindProc("SetEvent").Addr()
	cWaitForSingleObject = k.MustFindProc("WaitForSingleObject").Addr()
	a := syscall.MustLoadDLL("advapi32.dll")
	cRegisterServiceCtrlHandlerExW = a.MustFindProc("RegisterServiceCtrlHandlerExW").Addr()
}

// The HandlerEx prototype also has a context pointer but since we don't use
// it at start-up time we don't have to pass it over either.
type ctlEvent struct {
	cmd       Cmd
	eventType uint32
	eventData uintptr
	errno     uint32
}

// service provides access to windows service api.
type service struct {
	name    string
	h       windows.Handle
	cWaits  *event
	goWaits *event
	c       chan ctlEvent
	handler Handler
}

func newService(name string, handler Handler) (*service, error) {
	var s service
	var err error
	s.name = name
	s.c = make(chan ctlEvent)
	s.handler = handler
	s.cWaits, err = newEvent()
	if err != nil {
		return nil, err
	}
	s.goWaits, err = newEvent()
	if err != nil {
		s.cWaits.Close()
		return nil, err
	}
	return &s, nil
}

func (s *service) close() error {
	s.cWaits.Close()
	s.goWaits.Close()
	return nil
}

type exitCode struct {
	isSvcSpecific bool
	errno         uint32
}

func (s *service) updateStatus(status *Status, ec *exitCode) error {
	if s.h == 0 {
		return errors.New("updateStatus with no service status handle")
	}
	var t windows.SERVICE_STATUS
	t.ServiceType = windows.SERVICE_WIN32_OWN_PROCESS
	t.CurrentState = uint32(status.State)
	if status.Accepts&AcceptStop != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_STOP
	}
	if status.Accepts&AcceptShutdown != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_SHUTDOWN
	}
	if status.Accepts&AcceptPauseAndContinue != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_PAUSE_CONTINUE
	}
	if status.Accepts&AcceptParamChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_PARAMCHANGE
	}
	if status.Accepts&AcceptNetBindChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_NETBINDCHANGE
	}
	if status.Accepts&AcceptHardwareProfileChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_HARDWAREPROFILECHANGE
	}
	if status.Accepts&AcceptPowerEvent != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_POWEREVENT
	}
	if status.Accepts&AcceptSessionChange != 0 {
		t.ControlsAccepted |= windows.SERVICE_ACCEPT_SESSIONCHANGE
	}
	if ec.errno == 0 {
		t.Win32ExitCode = windows.NO_ERROR
		t.ServiceSpecificExitCode = windows.NO_ERROR
	} else if ec.isSvcSpecific {
		t.Win32ExitCode = uint32(windows.ERROR_SERVICE_SPECIFIC_ERROR)
		t.ServiceSpecificExitCode = ec.errno
	} else {
		t.Win32ExitCode = ec.errno
		t.ServiceSpecificExitCode = windows.NO_ERROR
	}
	t.CheckPoint = status.CheckPoint
	t.WaitHint = status.WaitHint
	return windows.SetServiceStatus(s.h, &t)
}

const (
	sysErrSetServiceStatusFailed = uint32(syscall.APPLICATION_ERROR) + iota
	sysErrNewThreadInCallback
)

func (s *service) run() {
	s.goWaits.Wait()
	s.h = windows.Handle(ssHandle)
	argv := (*[100]*int16)(unsafe.Pointer(sArgv))[:sArgc]
	args := make([]string, len(argv))
	for i, a := range argv {
		args[i] = syscall.UTF16ToString((*[1 << 20]uint16)(unsafe.Pointer(a))[:])
	}

	cmdsToHandler := make(chan ChangeRequest)
	changesFromHandler := make(chan Status)
	exitFromHandler := make(chan exitCode)

	go func() {
		ss, errno := s.handler.Execute(args, cmdsToHandler, changesFromHandler)
		exitFromHandler <- exitCode{ss, errno}
	}()

	status := Status{State: Stopped}
	ec := exitCode{isSvcSpecific: true, errno: 0}
	var outch chan ChangeRequest
	inch := s.c
	var cmd Cmd
	var evtype uint32
	var evdata uintptr
loop:
	for {
		select {
		case r := <-inch:
			if r.errno != 0 {
				ec.errno = r.errno
				break loop
			}
			inch = nil
			outch = cmdsToHandler
			cmd = r.cmd
			evtype = r.eventType
			evdata = r.eventData
		case outch <- ChangeRequest{cmd, evtype, evdata, status}:
			inch = s.c
			outch = nil
		case c := <-changesFromHandler:
			err := s.updateStatus(&c, &ec)
			if err != nil {
				// best suitable error number
				ec.errno = sysErrSetServiceStatusFailed
				if err2, ok := err.(syscall.Errno); ok {
					ec.errno = uint32(err2)
				}
				break loop
			}
			status = c
		case ec = <-exitFromHandler:
			break loop
		}
	}

	s.updateStatus(&Status{State: Stopped}, &ec)
	s.cWaits.Set()
}

func newCallback(fn interface{}) (cb uintptr, err error) {
	defer func() {
		r := recover()
		if r == nil {
			return
		}
		cb = 0
		switch v := r.(type) {
		case string:
			err = errors.New(v)
		case error:
			err = v
		default:
			err = errors.New("unexpected panic in syscall.NewCallback")
		}
	}()
	return syscall.NewCallback(fn), nil
}

// BUG(brainman): There is no mechanism to run multiple services
// inside one single executable. Perhaps, it can be overcome by
// using RegisterServiceCtrlHandlerEx Windows api.

// Run executes service name by calling appropriate handler function.
func Run(name string, handler Handler) error {
	runtime.LockOSThread()

	tid := windows.GetCurrentThreadId()

	s, err := newService(name, handler)
	if err != nil {
		return err
	}

	ctlHandler := func(ctl uint32, evtype uint32, evdata uintptr, context uintptr) uintptr {
		e := ctlEvent{cmd: Cmd(ctl), eventType: evtype, eventData: evdata}
		// We assume that this callback function is running on
		// the same thread as Run. Nowhere in MS documentation
		// I could find statement to guarantee that. So putting
		// check here to verify, otherwise things will go bad
		// quickly, if ignored.
		i := windows.GetCurrentThreadId()
		if i != tid {
			e.errno = sysErrNewThreadInCallback
		}
		s.c <- e
		// Always return NO_ERROR (0) for now.
		return 0
	}

	var svcmain uintptr
	getServiceMain(&svcmain)
	t := []windows.SERVICE_TABLE_ENTRY{
		{ServiceName: syscall.StringToUTF16Ptr(s.name), ServiceProc: svcmain},
		{ServiceName: nil, ServiceProc: 0},
	}

	goWaitsH = uintptr(s.goWaits.h)
	cWaitsH = uintptr(s.cWaits.h)
	sName = t[0].ServiceName
	ctlHandlerExProc, err = newCallback(ctlHandler)
	if err != nil {
		return err
	}

	go s.run()

	err = windows.StartServiceCtrlDispatcher(&t[0])
	if err != nil {
		return err
	}
	return nil
}

// StatusHandle returns service status handle. It is safe to call this function
// from inside the Handler.Execute because then it is guaranteed to be set.
// This code will have to change once multiple services are possible per process.
func StatusHandle() windows.Handle {
	return windows.Handle(ssHandle)
}
