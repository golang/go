// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"unsafe"
)

type Wndclassex struct {
	Size       uint32
	Style      uint32
	WndProc    uintptr
	ClsExtra   int32
	WndExtra   int32
	Instance   uint32
	Icon       uint32
	Cursor     uint32
	Background uint32
	MenuName   *uint16
	ClassName  *uint16
	IconSm     uint32
}

type Point struct {
	X int32
	Y int32
}

type Msg struct {
	Hwnd    uint32
	Message uint32
	Wparam  int32
	Lparam  int32
	Time    uint32
	Pt      Point
}

const (
	// Window styles
	WS_OVERLAPPED   = 0
	WS_POPUP        = 0x80000000
	WS_CHILD        = 0x40000000
	WS_MINIMIZE     = 0x20000000
	WS_VISIBLE      = 0x10000000
	WS_DISABLED     = 0x8000000
	WS_CLIPSIBLINGS = 0x4000000
	WS_CLIPCHILDREN = 0x2000000
	WS_MAXIMIZE     = 0x1000000
	WS_CAPTION      = WS_BORDER | WS_DLGFRAME
	WS_BORDER       = 0x800000
	WS_DLGFRAME     = 0x400000
	WS_VSCROLL      = 0x200000
	WS_HSCROLL      = 0x100000
	WS_SYSMENU      = 0x80000
	WS_THICKFRAME   = 0x40000
	WS_GROUP        = 0x20000
	WS_TABSTOP      = 0x10000
	WS_MINIMIZEBOX  = 0x20000
	WS_MAXIMIZEBOX  = 0x10000
	WS_TILED        = WS_OVERLAPPED
	WS_ICONIC       = WS_MINIMIZE
	WS_SIZEBOX      = WS_THICKFRAME
	// Common Window Styles
	WS_OVERLAPPEDWINDOW = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
	WS_TILEDWINDOW      = WS_OVERLAPPEDWINDOW
	WS_POPUPWINDOW      = WS_POPUP | WS_BORDER | WS_SYSMENU
	WS_CHILDWINDOW      = WS_CHILD

	WS_EX_CLIENTEDGE = 0x200

	// Some windows messages
	WM_CREATE  = 1
	WM_DESTROY = 2
	WM_CLOSE   = 16
	WM_COMMAND = 273

	// Some button control styles
	BS_DEFPUSHBUTTON = 1

	// Some color constants
	COLOR_WINDOW  = 5
	COLOR_BTNFACE = 15

	// Default window position
	CW_USEDEFAULT = 0x80000000 - 0x100000000

	// Show window default style
	SW_SHOWDEFAULT = 10
)

var (
	// Some globally known cursors
	IDC_ARROW = MakeIntResource(32512)
	IDC_IBEAM = MakeIntResource(32513)
	IDC_WAIT  = MakeIntResource(32514)
	IDC_CROSS = MakeIntResource(32515)

	// Some globally known icons
	IDI_APPLICATION = MakeIntResource(32512)
	IDI_HAND        = MakeIntResource(32513)
	IDI_QUESTION    = MakeIntResource(32514)
	IDI_EXCLAMATION = MakeIntResource(32515)
	IDI_ASTERISK    = MakeIntResource(32516)
	IDI_WINLOGO     = MakeIntResource(32517)
	IDI_WARNING     = IDI_EXCLAMATION
	IDI_ERROR       = IDI_HAND
	IDI_INFORMATION = IDI_ASTERISK
)

//sys	GetModuleHandle(modname *uint16) (handle uint32, errno int) = GetModuleHandleW
//sys	RegisterClassEx(wndclass *Wndclassex) (atom uint16, errno int) = user32.RegisterClassExW
//sys	CreateWindowEx(exstyle uint32, classname *uint16, windowname *uint16, style uint32, x int32, y int32, width int32, height int32, wndparent uint32, menu uint32, instance uint32, param uintptr) (hwnd uint32, errno int) = user32.CreateWindowExW
//sys	DefWindowProc(hwnd uint32, msg uint32, wparam int32, lparam int32) (lresult int32) = user32.DefWindowProcW
//sys	DestroyWindow(hwnd uint32) (errno int) = user32.DestroyWindow
//sys	PostQuitMessage(exitcode int32) = user32.PostQuitMessage
//sys	ShowWindow(hwnd uint32, cmdshow int32) (wasvisible bool) = user32.ShowWindow
//sys	UpdateWindow(hwnd uint32) (errno int) = user32.UpdateWindow
//sys	GetMessage(msg *Msg, hwnd uint32, MsgFilterMin uint32, MsgFilterMax uint32) (ret int32, errno int) [failretval==-1] = user32.GetMessageW
//sys	TranslateMessage(msg *Msg) (done bool) = user32.TranslateMessage
//sys	DispatchMessage(msg *Msg) (ret int32) = user32.DispatchMessageW
//sys	LoadIcon(instance uint32, iconname *uint16) (icon uint32, errno int) = user32.LoadIconW
//sys	LoadCursor(instance uint32, cursorname *uint16) (cursor uint32, errno int) = user32.LoadCursorW
//sys	SetCursor(cursor uint32) (precursor uint32, errno int) = user32.SetCursor
//sys	SendMessage(hwnd uint32, msg uint32, wparam int32, lparam int32) (lresult int32) = user32.SendMessageW
//sys	PostMessage(hwnd uint32, msg uint32, wparam int32, lparam int32) (errno int) = user32.PostMessageW

func MakeIntResource(id uint16) *uint16 {
	return (*uint16)(unsafe.Pointer(uintptr(id)))
}
