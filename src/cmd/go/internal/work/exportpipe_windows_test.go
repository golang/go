// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !cmd_go_bootstrap && windows

package work

import (
	"path/filepath"
	"runtime"
	"testing"
	"unsafe"

	"golang.org/x/sys/windows"
)

func TestExportPipeSecurityDescriptor(t *testing.T) {
	objdir := t.TempDir() + string(filepath.Separator)
	ep, err := newExportPipe(objdir)
	if err != nil {
		t.Fatalf("newExportPipe: %v", err)
	}
	defer ep.close()

	user, err := windows.GetCurrentProcessToken().GetTokenUser()
	if err != nil {
		t.Fatalf("GetTokenUser: %v", err)
	}

	sd, err := windows.GetSecurityInfo(ep.h, windows.SE_KERNEL_OBJECT, windows.DACL_SECURITY_INFORMATION)
	if err != nil {
		t.Fatalf("GetSecurityInfo: %v", err)
	}
	defer runtime.KeepAlive(sd)

	control, _, err := sd.Control()
	if err != nil {
		t.Fatalf("Control: %v", err)
	}
	if control&windows.SE_DACL_PROTECTED == 0 {
		t.Errorf("security descriptor DACL is not protected: control=%#x, sddl=%q", control, sd)
	}
	dacl, defaulted, err := sd.DACL()
	if err != nil {
		t.Fatalf("DACL: %v", err)
	}
	if defaulted {
		t.Errorf("security descriptor DACL is defaulted: sddl=%q", sd)
	}
	if dacl == nil {
		t.Fatalf("DACL is nil (sddl=%q)", sd)
	}
	if dacl.AceCount != 1 {
		t.Fatalf("DACL AceCount = %d, want 1 (sddl=%q)", dacl.AceCount, sd)
	}
	var ace *windows.ACCESS_ALLOWED_ACE
	if err := windows.GetAce(dacl, 0, &ace); err != nil {
		t.Fatalf("GetAce: %v", err)
	}
	if ace.Header.AceType != windows.ACCESS_ALLOWED_ACE_TYPE {
		t.Errorf("ACE type = %d, want ACCESS_ALLOWED_ACE_TYPE (%d)", ace.Header.AceType, windows.ACCESS_ALLOWED_ACE_TYPE)
	}
	sid := (*windows.SID)(unsafe.Pointer(&ace.SidStart))
	if !sid.Equals(user.User.Sid) {
		t.Errorf("ACE SID = %s, want current user SID %s", sid, user.User.Sid)
	}
}
