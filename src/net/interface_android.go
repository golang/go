// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"os"
	"syscall"
	"unsafe"
)

type ifReq [40]byte

// Starting from Android 11, it is no longer possible to retrieve network card information
// using the RTM_GETLINK method.
// As a result, alternative methods need to be employed.
// After considering the Android NetworkInterface.getNetworkInterfaces() method,
// I opted to utilize the RTM_GETADDR + ioctl approach to obtain network card information.
// However, it appears that retrieving the
// HWAddr (hardware address) of the network card is currently not achievable.
func interfaceTableAndroid(ifindex int) ([]Interface, error) {
	tab, err := syscall.NetlinkRIB(syscall.RTM_GETADDR, syscall.AF_UNSPEC)
	if err != nil {
		return nil, os.NewSyscallError("netlinkrib", err)
	}
	msgs, err := syscall.ParseNetlinkMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("parsenetlinkmessage", err)
	}

	var ift []Interface
	im := make(map[uint32]struct{})
loop:
	for _, m := range msgs {
		switch m.Header.Type {
		case syscall.NLMSG_DONE:
			break loop
		case syscall.RTM_NEWADDR:
			ifam := (*syscall.IfAddrmsg)(unsafe.Pointer(&m.Data[0]))
			if _, ok := im[ifam.Index]; ok {
				continue
			} else {
				im[ifam.Index] = struct{}{}
			}

			if ifindex == 0 || ifindex == int(ifam.Index) {
				ifi := newLinkAndroid(ifam)
				if ifi != nil {
					ift = append(ift, *ifi)
				}
				if ifindex == int(ifam.Index) {
					break loop
				}
			}
		}
	}

	return ift, nil
}

// According to the network card Index, get the Name, MTU and Flags of the network card through ioctl
func newLinkAndroid(ifam *syscall.IfAddrmsg) *Interface {
	ift := &Interface{Index: int(ifam.Index)}

	name, err := indexToName(ifam.Index)
	if err != nil {
		return nil
	}
	ift.Name = name

	mtu, err := nameToMTU(name)
	if err != nil {
		return nil
	}
	ift.MTU = mtu

	flags, err := nameToFlags(name)
	if err != nil {
		return nil
	}
	ift.Flags = flags
	return ift
}

func ioctl(fd int, req uint, arg unsafe.Pointer) error {
	_, _, e1 := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), uintptr(req), uintptr(arg))
	if e1 != 0 {
		return e1
	}
	return nil
}

func indexToName(index uint32) (string, error) {
	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return "", err
	}
	defer syscall.Close(fd)

	var ifr ifReq
	*(*uint32)(unsafe.Pointer(&ifr[syscall.IFNAMSIZ])) = index
	err = ioctl(fd, syscall.SIOCGIFNAME, unsafe.Pointer(&ifr[0]))
	if err != nil {
		return "", err
	}

	return string(bytes.Trim(ifr[:syscall.IFNAMSIZ], "\x00")), nil
}

func nameToMTU(name string) (int, error) {
	// Leave room for terminating NULL byte.
	if len(name) >= syscall.IFNAMSIZ {
		return 0, syscall.EINVAL
	}

	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return 0, err
	}
	defer syscall.Close(fd)

	var ifr ifReq
	copy(ifr[:], name)
	err = ioctl(fd, syscall.SIOCGIFMTU, unsafe.Pointer(&ifr[0]))
	if err != nil {
		return 0, err
	}

	return int(*(*int32)(unsafe.Pointer(&ifr[syscall.IFNAMSIZ]))), nil
}

func nameToFlags(name string) (Flags, error) {
	// Leave room for terminating NULL byte.
	if len(name) >= syscall.IFNAMSIZ {
		return 0, syscall.EINVAL
	}

	fd, err := syscall.Socket(syscall.AF_INET, syscall.SOCK_DGRAM|syscall.SOCK_CLOEXEC, 0)
	if err != nil {
		return 0, err
	}
	defer syscall.Close(fd)

	var ifr ifReq
	copy(ifr[:], name)
	err = ioctl(fd, syscall.SIOCGIFFLAGS, unsafe.Pointer(&ifr[0]))
	if err != nil {
		return 0, err
	}

	return linkFlags(*(*uint32)(unsafe.Pointer(&ifr[syscall.IFNAMSIZ]))), nil
}
