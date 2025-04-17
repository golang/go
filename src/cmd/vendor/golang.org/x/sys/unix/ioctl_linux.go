// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import "unsafe"

// IoctlRetInt performs an ioctl operation specified by req on a device
// associated with opened file descriptor fd, and returns a non-negative
// integer that is returned by the ioctl syscall.
func IoctlRetInt(fd int, req uint) (int, error) {
	ret, _, err := Syscall(SYS_IOCTL, uintptr(fd), uintptr(req), 0)
	if err != 0 {
		return 0, err
	}
	return int(ret), nil
}

func IoctlGetUint32(fd int, req uint) (uint32, error) {
	var value uint32
	err := ioctlPtr(fd, req, unsafe.Pointer(&value))
	return value, err
}

func IoctlGetRTCTime(fd int) (*RTCTime, error) {
	var value RTCTime
	err := ioctlPtr(fd, RTC_RD_TIME, unsafe.Pointer(&value))
	return &value, err
}

func IoctlSetRTCTime(fd int, value *RTCTime) error {
	return ioctlPtr(fd, RTC_SET_TIME, unsafe.Pointer(value))
}

func IoctlGetRTCWkAlrm(fd int) (*RTCWkAlrm, error) {
	var value RTCWkAlrm
	err := ioctlPtr(fd, RTC_WKALM_RD, unsafe.Pointer(&value))
	return &value, err
}

func IoctlSetRTCWkAlrm(fd int, value *RTCWkAlrm) error {
	return ioctlPtr(fd, RTC_WKALM_SET, unsafe.Pointer(value))
}

// IoctlGetEthtoolDrvinfo fetches ethtool driver information for the network
// device specified by ifname.
func IoctlGetEthtoolDrvinfo(fd int, ifname string) (*EthtoolDrvinfo, error) {
	ifr, err := NewIfreq(ifname)
	if err != nil {
		return nil, err
	}

	value := EthtoolDrvinfo{Cmd: ETHTOOL_GDRVINFO}
	ifrd := ifr.withData(unsafe.Pointer(&value))

	err = ioctlIfreqData(fd, SIOCETHTOOL, &ifrd)
	return &value, err
}

// IoctlGetEthtoolTsInfo fetches ethtool timestamping and PHC
// association for the network device specified by ifname.
func IoctlGetEthtoolTsInfo(fd int, ifname string) (*EthtoolTsInfo, error) {
	ifr, err := NewIfreq(ifname)
	if err != nil {
		return nil, err
	}

	value := EthtoolTsInfo{Cmd: ETHTOOL_GET_TS_INFO}
	ifrd := ifr.withData(unsafe.Pointer(&value))

	err = ioctlIfreqData(fd, SIOCETHTOOL, &ifrd)
	return &value, err
}

// IoctlGetHwTstamp retrieves the hardware timestamping configuration
// for the network device specified by ifname.
func IoctlGetHwTstamp(fd int, ifname string) (*HwTstampConfig, error) {
	ifr, err := NewIfreq(ifname)
	if err != nil {
		return nil, err
	}

	value := HwTstampConfig{}
	ifrd := ifr.withData(unsafe.Pointer(&value))

	err = ioctlIfreqData(fd, SIOCGHWTSTAMP, &ifrd)
	return &value, err
}

// IoctlSetHwTstamp updates the hardware timestamping configuration for
// the network device specified by ifname.
func IoctlSetHwTstamp(fd int, ifname string, cfg *HwTstampConfig) error {
	ifr, err := NewIfreq(ifname)
	if err != nil {
		return err
	}
	ifrd := ifr.withData(unsafe.Pointer(cfg))
	return ioctlIfreqData(fd, SIOCSHWTSTAMP, &ifrd)
}

// FdToClockID derives the clock ID from the file descriptor number
// - see clock_gettime(3), FD_TO_CLOCKID macros. The resulting ID is
// suitable for system calls like ClockGettime.
func FdToClockID(fd int) int32 { return int32((int(^fd) << 3) | 3) }

// IoctlPtpClockGetcaps returns the description of a given PTP device.
func IoctlPtpClockGetcaps(fd int) (*PtpClockCaps, error) {
	var value PtpClockCaps
	err := ioctlPtr(fd, PTP_CLOCK_GETCAPS2, unsafe.Pointer(&value))
	return &value, err
}

// IoctlPtpSysOffsetPrecise returns a description of the clock
// offset compared to the system clock.
func IoctlPtpSysOffsetPrecise(fd int) (*PtpSysOffsetPrecise, error) {
	var value PtpSysOffsetPrecise
	err := ioctlPtr(fd, PTP_SYS_OFFSET_PRECISE2, unsafe.Pointer(&value))
	return &value, err
}

// IoctlPtpSysOffsetExtended returns an extended description of the
// clock offset compared to the system clock. The samples parameter
// specifies the desired number of measurements.
func IoctlPtpSysOffsetExtended(fd int, samples uint) (*PtpSysOffsetExtended, error) {
	value := PtpSysOffsetExtended{Samples: uint32(samples)}
	err := ioctlPtr(fd, PTP_SYS_OFFSET_EXTENDED2, unsafe.Pointer(&value))
	return &value, err
}

// IoctlPtpPinGetfunc returns the configuration of the specified
// I/O pin on given PTP device.
func IoctlPtpPinGetfunc(fd int, index uint) (*PtpPinDesc, error) {
	value := PtpPinDesc{Index: uint32(index)}
	err := ioctlPtr(fd, PTP_PIN_GETFUNC2, unsafe.Pointer(&value))
	return &value, err
}

// IoctlPtpPinSetfunc updates configuration of the specified PTP
// I/O pin.
func IoctlPtpPinSetfunc(fd int, pd *PtpPinDesc) error {
	return ioctlPtr(fd, PTP_PIN_SETFUNC2, unsafe.Pointer(pd))
}

// IoctlPtpPeroutRequest configures the periodic output mode of the
// PTP I/O pins.
func IoctlPtpPeroutRequest(fd int, r *PtpPeroutRequest) error {
	return ioctlPtr(fd, PTP_PEROUT_REQUEST2, unsafe.Pointer(r))
}

// IoctlPtpExttsRequest configures the external timestamping mode
// of the PTP I/O pins.
func IoctlPtpExttsRequest(fd int, r *PtpExttsRequest) error {
	return ioctlPtr(fd, PTP_EXTTS_REQUEST2, unsafe.Pointer(r))
}

// IoctlGetWatchdogInfo fetches information about a watchdog device from the
// Linux watchdog API. For more information, see:
// https://www.kernel.org/doc/html/latest/watchdog/watchdog-api.html.
func IoctlGetWatchdogInfo(fd int) (*WatchdogInfo, error) {
	var value WatchdogInfo
	err := ioctlPtr(fd, WDIOC_GETSUPPORT, unsafe.Pointer(&value))
	return &value, err
}

// IoctlWatchdogKeepalive issues a keepalive ioctl to a watchdog device. For
// more information, see:
// https://www.kernel.org/doc/html/latest/watchdog/watchdog-api.html.
func IoctlWatchdogKeepalive(fd int) error {
	// arg is ignored and not a pointer, so ioctl is fine instead of ioctlPtr.
	return ioctl(fd, WDIOC_KEEPALIVE, 0)
}

// IoctlFileCloneRange performs an FICLONERANGE ioctl operation to clone the
// range of data conveyed in value to the file associated with the file
// descriptor destFd. See the ioctl_ficlonerange(2) man page for details.
func IoctlFileCloneRange(destFd int, value *FileCloneRange) error {
	return ioctlPtr(destFd, FICLONERANGE, unsafe.Pointer(value))
}

// IoctlFileClone performs an FICLONE ioctl operation to clone the entire file
// associated with the file description srcFd to the file associated with the
// file descriptor destFd. See the ioctl_ficlone(2) man page for details.
func IoctlFileClone(destFd, srcFd int) error {
	return ioctl(destFd, FICLONE, uintptr(srcFd))
}

type FileDedupeRange struct {
	Src_offset uint64
	Src_length uint64
	Reserved1  uint16
	Reserved2  uint32
	Info       []FileDedupeRangeInfo
}

type FileDedupeRangeInfo struct {
	Dest_fd       int64
	Dest_offset   uint64
	Bytes_deduped uint64
	Status        int32
	Reserved      uint32
}

// IoctlFileDedupeRange performs an FIDEDUPERANGE ioctl operation to share the
// range of data conveyed in value from the file associated with the file
// descriptor srcFd to the value.Info destinations. See the
// ioctl_fideduperange(2) man page for details.
func IoctlFileDedupeRange(srcFd int, value *FileDedupeRange) error {
	buf := make([]byte, SizeofRawFileDedupeRange+
		len(value.Info)*SizeofRawFileDedupeRangeInfo)
	rawrange := (*RawFileDedupeRange)(unsafe.Pointer(&buf[0]))
	rawrange.Src_offset = value.Src_offset
	rawrange.Src_length = value.Src_length
	rawrange.Dest_count = uint16(len(value.Info))
	rawrange.Reserved1 = value.Reserved1
	rawrange.Reserved2 = value.Reserved2

	for i := range value.Info {
		rawinfo := (*RawFileDedupeRangeInfo)(unsafe.Pointer(
			uintptr(unsafe.Pointer(&buf[0])) + uintptr(SizeofRawFileDedupeRange) +
				uintptr(i*SizeofRawFileDedupeRangeInfo)))
		rawinfo.Dest_fd = value.Info[i].Dest_fd
		rawinfo.Dest_offset = value.Info[i].Dest_offset
		rawinfo.Bytes_deduped = value.Info[i].Bytes_deduped
		rawinfo.Status = value.Info[i].Status
		rawinfo.Reserved = value.Info[i].Reserved
	}

	err := ioctlPtr(srcFd, FIDEDUPERANGE, unsafe.Pointer(&buf[0]))

	// Output
	for i := range value.Info {
		rawinfo := (*RawFileDedupeRangeInfo)(unsafe.Pointer(
			uintptr(unsafe.Pointer(&buf[0])) + uintptr(SizeofRawFileDedupeRange) +
				uintptr(i*SizeofRawFileDedupeRangeInfo)))
		value.Info[i].Dest_fd = rawinfo.Dest_fd
		value.Info[i].Dest_offset = rawinfo.Dest_offset
		value.Info[i].Bytes_deduped = rawinfo.Bytes_deduped
		value.Info[i].Status = rawinfo.Status
		value.Info[i].Reserved = rawinfo.Reserved
	}

	return err
}

func IoctlHIDGetDesc(fd int, value *HIDRawReportDescriptor) error {
	return ioctlPtr(fd, HIDIOCGRDESC, unsafe.Pointer(value))
}

func IoctlHIDGetRawInfo(fd int) (*HIDRawDevInfo, error) {
	var value HIDRawDevInfo
	err := ioctlPtr(fd, HIDIOCGRAWINFO, unsafe.Pointer(&value))
	return &value, err
}

func IoctlHIDGetRawName(fd int) (string, error) {
	var value [_HIDIOCGRAWNAME_LEN]byte
	err := ioctlPtr(fd, _HIDIOCGRAWNAME, unsafe.Pointer(&value[0]))
	return ByteSliceToString(value[:]), err
}

func IoctlHIDGetRawPhys(fd int) (string, error) {
	var value [_HIDIOCGRAWPHYS_LEN]byte
	err := ioctlPtr(fd, _HIDIOCGRAWPHYS, unsafe.Pointer(&value[0]))
	return ByteSliceToString(value[:]), err
}

func IoctlHIDGetRawUniq(fd int) (string, error) {
	var value [_HIDIOCGRAWUNIQ_LEN]byte
	err := ioctlPtr(fd, _HIDIOCGRAWUNIQ, unsafe.Pointer(&value[0]))
	return ByteSliceToString(value[:]), err
}

// IoctlIfreq performs an ioctl using an Ifreq structure for input and/or
// output. See the netdevice(7) man page for details.
func IoctlIfreq(fd int, req uint, value *Ifreq) error {
	// It is possible we will add more fields to *Ifreq itself later to prevent
	// misuse, so pass the raw *ifreq directly.
	return ioctlPtr(fd, req, unsafe.Pointer(&value.raw))
}

// TODO(mdlayher): export if and when IfreqData is exported.

// ioctlIfreqData performs an ioctl using an ifreqData structure for input
// and/or output. See the netdevice(7) man page for details.
func ioctlIfreqData(fd int, req uint, value *ifreqData) error {
	// The memory layout of IfreqData (type-safe) and ifreq (not type-safe) are
	// identical so pass *IfreqData directly.
	return ioctlPtr(fd, req, unsafe.Pointer(value))
}

// IoctlKCMClone attaches a new file descriptor to a multiplexor by cloning an
// existing KCM socket, returning a structure containing the file descriptor of
// the new socket.
func IoctlKCMClone(fd int) (*KCMClone, error) {
	var info KCMClone
	if err := ioctlPtr(fd, SIOCKCMCLONE, unsafe.Pointer(&info)); err != nil {
		return nil, err
	}

	return &info, nil
}

// IoctlKCMAttach attaches a TCP socket and associated BPF program file
// descriptor to a multiplexor.
func IoctlKCMAttach(fd int, info KCMAttach) error {
	return ioctlPtr(fd, SIOCKCMATTACH, unsafe.Pointer(&info))
}

// IoctlKCMUnattach unattaches a TCP socket file descriptor from a multiplexor.
func IoctlKCMUnattach(fd int, info KCMUnattach) error {
	return ioctlPtr(fd, SIOCKCMUNATTACH, unsafe.Pointer(&info))
}

// IoctlLoopGetStatus64 gets the status of the loop device associated with the
// file descriptor fd using the LOOP_GET_STATUS64 operation.
func IoctlLoopGetStatus64(fd int) (*LoopInfo64, error) {
	var value LoopInfo64
	if err := ioctlPtr(fd, LOOP_GET_STATUS64, unsafe.Pointer(&value)); err != nil {
		return nil, err
	}
	return &value, nil
}

// IoctlLoopSetStatus64 sets the status of the loop device associated with the
// file descriptor fd using the LOOP_SET_STATUS64 operation.
func IoctlLoopSetStatus64(fd int, value *LoopInfo64) error {
	return ioctlPtr(fd, LOOP_SET_STATUS64, unsafe.Pointer(value))
}

// IoctlLoopConfigure configures all loop device parameters in a single step
func IoctlLoopConfigure(fd int, value *LoopConfig) error {
	return ioctlPtr(fd, LOOP_CONFIGURE, unsafe.Pointer(value))
}
