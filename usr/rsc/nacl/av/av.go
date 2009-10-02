// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Native Client audio/video

// Package av implements audio and video access for Native Client
// binaries running standalone or embedded in a web browser window.
package av

import (
	"bytes";
	"draw";
	"log";
	"nacl/srpc";
	"os";
	"syscall";
	"unsafe";
)

var srpcEnabled = srpc.Enabled();

// native_client/src/trusted/service_runtime/include/sys/audio_video.h

// Subsystem values for Init.
const (
	SubsystemVideo = 1<<iota;
	SubsystemAudio;
	SubsystemEmbed;
)
//	SubsystemRawEvents;

// Audio formats.
const (
	AudioFormatStereo44K = iota;
	AudioFormatStereo48K;
)

// A Window represents a connection to the Native Client window.
// It implements draw.Context.
type Window struct {
	Embedded bool;	// running as part of a web page?
	*Image;		// screen image

	mousec chan draw.Mouse;
	kbdc chan int;
	quitc chan bool;
	resizec chan bool;
}

// *Window implements draw.Context
var _ draw.Context = (*Window)(nil)

func (w *Window) KeyboardChan() <-chan int {
	return w.kbdc;
}

func (w *Window) MouseChan() <-chan draw.Mouse {
	return w.mousec;
}

func (w *Window) QuitChan() <-chan bool {
	return w.quitc;
}

func (w *Window) ResizeChan() <-chan bool {
	return w.resizec;
}

func (w *Window) Screen() draw.Image {
	return w.Image;
}

// Init initializes the Native Client subsystems specified by subsys.
// Init must be called before using any of the other functions
// in this package, and it must be called only once.
//
// If the SubsystemVideo flag is set, Init requests a window of size dx×dy.
// When embedded in a web page, the web page's window specification
// overrides the parameters to Init, so the returned Window may have
// a different size than requested.
//
// If the SubsystemAudio flag is set, Init requests a connection to the
// audio device carrying 44 kHz 16-bit stereo PCM audio samples.
func Init(subsys int, dx, dy int) (*Window, os.Error) {
	xsubsys := subsys;
	if srpcEnabled {
		waitBridge();
		xsubsys &^= SubsystemVideo|SubsystemEmbed;
	}

	if xsubsys & SubsystemEmbed != 0 {
		return nil, os.NewError("not embedded");
	}

	w := new(Window);
	err := multimediaInit(xsubsys);
	if err != nil {
		return nil, err;
	}

	if subsys&SubsystemVideo != 0 {
		if dx, dy, err = videoInit(dx, dy); err != nil {
			return nil, err;
		}
		w.Image = newImage(dx, dy, bridge.pixel);
		w.resizec = make(chan bool, 64);
		w.kbdc = make(chan int, 64);
		w.mousec = make(chan draw.Mouse, 64);
		w.quitc = make(chan bool);
	}

	if subsys&SubsystemAudio != 0 {
		var n int;
		if n, err = audioInit(AudioFormatStereo44K, 2048); err != nil {
			return nil, err;
		}
		println("audio", n);
	}

	if subsys&SubsystemVideo != 0 {
		go w.readEvents();
	}

	return w, nil;
}

func (w *Window) FlushImage() {
	if w.Image == nil {
		return;
	}
	videoUpdate(w.Image.Linear);
}

func multimediaInit(subsys int) (err os.Error) {
	return os.NewSyscallError("multimedia_init", syscall.MultimediaInit(subsys));
}

func videoInit(dx, dy int) (ndx, ndy int, err os.Error) {
	if srpcEnabled {
		bridge.share.ready = 1;
		return int(bridge.share.width), int(bridge.share.height), nil;
	}
	if e := syscall.VideoInit(dx, dy); e != 0 {
		return 0, 0, os.NewSyscallError("video_init", int(e));
	}
	return dx, dy, nil;
}

func videoUpdate(data []Color) (err os.Error) {
	if srpcEnabled {
		bridge.flushRPC.Call("upcall", nil);
		return;
	}
	return os.NewSyscallError("video_update", syscall.VideoUpdate((*uint32)(&data[0])));
}

var noEvents = os.NewError("no events");

func videoPollEvent(ev []byte) (err os.Error) {
	if srpcEnabled {
		r := bridge.share.eq.ri;
		if r == bridge.share.eq.wi {
			return noEvents;
		}
		bytes.Copy(ev, &bridge.share.eq.event[r]);
		bridge.share.eq.ri = (r+1) % eqsize;
		return nil;
	}
	return os.NewSyscallError("video_poll_event", syscall.VideoPollEvent(&ev[0]));
}

func audioInit(fmt int, want int) (got int, err os.Error) {
	var x int;
	e := syscall.AudioInit(fmt, want, &x);
	if e == 0 {
		return x, nil;
	}
	return 0, os.NewSyscallError("audio_init", e);
}

var audioSize uintptr

// AudioStream provides access to the audio device.
// Each call to AudioStream writes the given data,
// which should be a slice of 16-bit stereo PCM audio samples,
// and returns the number of samples required by the next
// call to AudioStream.
//
// To find out the initial number of samples to write, call AudioStream(nil).
//
func AudioStream(data []uint16) (nextSize int, err os.Error) {
	if audioSize == 0 {
		e := os.NewSyscallError("audio_stream", syscall.AudioStream(nil, &audioSize));
		return int(audioSize), e;
	}
	if data == nil {
		return int(audioSize), nil;
	}
	if uintptr(len(data))*2 != audioSize {
		log.Stdoutf("invalid audio size want %d got %d", audioSize, len(data));
	}
	e := os.NewSyscallError("audio_stream", syscall.AudioStream(&data[0], &audioSize));
	return int(audioSize), e;
}

// Synchronization structure to wait for bridge to become ready.
var bridge struct {
	c chan bool;
	displayFd int;
	rpcFd int;
	share *videoShare;
	pixel []Color;
	client *srpc.Client;
	flushRPC *srpc.RPC;
}

// Wait for bridge to become ready.
// When chan is first created, there is nothing in it,
// so this blocks.  Once the bridge is ready, multimediaBridge.Run
// will drop a value into the channel.  Then any calls
// to waitBridge will finish, taking the value out and immediately putting it back.
func waitBridge() {
	bridge.c <- <-bridge.c;
}

const eqsize = 64;

// Data structure shared with host via mmap.
type videoShare struct {
	revision int32;	// definition below is rev 100 unless noted
	mapSize int32;

	// event queue
	eq struct {
		ri uint32;	// read index [0,eqsize)
		wi uint32;	// write index [0,eqsize)
		eof int32;
		event [eqsize][64]byte;
	};

	// now unused
	_, _, _, _ int32;

	// video backing store information
	width, height, _, size int32;
	ready int32;	// rev 0x101
}

// The frame buffer data is videoShareSize bytes after
// the videoShare begins.
const videoShareSize = 16*1024

type multimediaBridge struct{}

// If using SRPC, the runtime will call this method to pass in two file descriptors,
// one to mmap to get the display memory, and another to use for SRPCs back
// to the main process.
func (multimediaBridge) Run(arg, ret []interface{}, size []int) srpc.Errno {
	bridge.displayFd = arg[0].(int);
	bridge.rpcFd = arg[1].(int);

	var st syscall.Stat_t;
	if errno := syscall.Fstat(bridge.displayFd, &st); errno != 0 {
		log.Exitf("mmbridge stat display: %s", os.Errno(errno));
	}

	addr, _, errno := syscall.Syscall6(syscall.SYS_MMAP,
		0,
		uintptr(st.Size),
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
		uintptr(bridge.displayFd),
		0);
	if errno != 0 {
		log.Exitf("mmap display: %s", os.Errno(errno));
	}

	bridge.share = (*videoShare)(unsafe.Pointer(addr));

	// Overestimate frame buffer size
	// (must use a compile-time constant)
	// and then reslice.  256 megapixels (1 GB) should be enough.
	fb := (*[256*1024*1024]Color)(unsafe.Pointer(addr+videoShareSize));
	bridge.pixel = fb[0:(st.Size - videoShareSize)/4];

	// Configure RPC connection back to client.
	var err os.Error;
	bridge.client, err = srpc.NewClient(bridge.rpcFd);
	if err != nil {
		log.Exitf("NewClient: %s", err);
	}
	bridge.flushRPC = bridge.client.NewRPC(nil);

	// Notify waiters that the bridge is ready.
	println("bridged", bridge.share.revision);
	bridge.c <- true;

	return srpc.OK;
}

func init() {
	bridge.c = make(chan bool, 1);
	if srpcEnabled {
		srpc.Add("nacl_multimedia_bridge", "hh:", multimediaBridge{});
	}
}

