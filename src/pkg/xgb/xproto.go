// This file was generated automatically from xproto.xml.

package xgb

import "os"

type Char2b struct {
	Byte1 byte
	Byte2 byte
}

func getChar2b(b []byte, v *Char2b) int {
	v.Byte1 = b[0]
	v.Byte2 = b[1]
	return 2
}

func (c *Conn) sendChar2bList(list []Char2b, count int) {
	b0 := make([]byte, 2*count)
	for k := 0; k < count; k++ {
		b := b0[k*2:]
		b[0] = list[k].Byte1
		b[1] = list[k].Byte2
	}
	c.sendBytes(b0)
}

type Point struct {
	X int16
	Y int16
}

func getPoint(b []byte, v *Point) int {
	v.X = int16(get16(b[0:]))
	v.Y = int16(get16(b[2:]))
	return 4
}

func (c *Conn) sendPointList(list []Point, count int) {
	b0 := make([]byte, 4*count)
	for k := 0; k < count; k++ {
		b := b0[k*4:]
		put16(b[0:], uint16(list[k].X))
		put16(b[2:], uint16(list[k].Y))
	}
	c.sendBytes(b0)
}

type Rectangle struct {
	X      int16
	Y      int16
	Width  uint16
	Height uint16
}

func getRectangle(b []byte, v *Rectangle) int {
	v.X = int16(get16(b[0:]))
	v.Y = int16(get16(b[2:]))
	v.Width = get16(b[4:])
	v.Height = get16(b[6:])
	return 8
}

func (c *Conn) sendRectangleList(list []Rectangle, count int) {
	b0 := make([]byte, 8*count)
	for k := 0; k < count; k++ {
		b := b0[k*8:]
		put16(b[0:], uint16(list[k].X))
		put16(b[2:], uint16(list[k].Y))
		put16(b[4:], list[k].Width)
		put16(b[6:], list[k].Height)
	}
	c.sendBytes(b0)
}

type Arc struct {
	X      int16
	Y      int16
	Width  uint16
	Height uint16
	Angle1 int16
	Angle2 int16
}

func getArc(b []byte, v *Arc) int {
	v.X = int16(get16(b[0:]))
	v.Y = int16(get16(b[2:]))
	v.Width = get16(b[4:])
	v.Height = get16(b[6:])
	v.Angle1 = int16(get16(b[8:]))
	v.Angle2 = int16(get16(b[10:]))
	return 12
}

func (c *Conn) sendArcList(list []Arc, count int) {
	b0 := make([]byte, 12*count)
	for k := 0; k < count; k++ {
		b := b0[k*12:]
		put16(b[0:], uint16(list[k].X))
		put16(b[2:], uint16(list[k].Y))
		put16(b[4:], list[k].Width)
		put16(b[6:], list[k].Height)
		put16(b[8:], uint16(list[k].Angle1))
		put16(b[10:], uint16(list[k].Angle2))
	}
	c.sendBytes(b0)
}

type Format struct {
	Depth        byte
	BitsPerPixel byte
	ScanlinePad  byte
}

func getFormat(b []byte, v *Format) int {
	v.Depth = b[0]
	v.BitsPerPixel = b[1]
	v.ScanlinePad = b[2]
	return 8
}

const (
	VisualClassStaticGray  = 0
	VisualClassGrayScale   = 1
	VisualClassStaticColor = 2
	VisualClassPseudoColor = 3
	VisualClassTrueColor   = 4
	VisualClassDirectColor = 5
)

type VisualInfo struct {
	VisualId        Id
	Class           byte
	BitsPerRgbValue byte
	ColormapEntries uint16
	RedMask         uint32
	GreenMask       uint32
	BlueMask        uint32
}

func getVisualInfo(b []byte, v *VisualInfo) int {
	v.VisualId = Id(get32(b[0:]))
	v.Class = b[4]
	v.BitsPerRgbValue = b[5]
	v.ColormapEntries = get16(b[6:])
	v.RedMask = get32(b[8:])
	v.GreenMask = get32(b[12:])
	v.BlueMask = get32(b[16:])
	return 24
}

type DepthInfo struct {
	Depth      byte
	VisualsLen uint16
	Visuals    []VisualInfo
}

func getDepthInfo(b []byte, v *DepthInfo) int {
	v.Depth = b[0]
	v.VisualsLen = get16(b[2:])
	offset := 8
	v.Visuals = make([]VisualInfo, int(v.VisualsLen))
	for i := 0; i < int(v.VisualsLen); i++ {
		offset += getVisualInfo(b[offset:], &v.Visuals[i])
	}
	return offset
}

const (
	EventMaskNoEvent              = 0
	EventMaskKeyPress             = 1
	EventMaskKeyRelease           = 2
	EventMaskButtonPress          = 4
	EventMaskButtonRelease        = 8
	EventMaskEnterWindow          = 16
	EventMaskLeaveWindow          = 32
	EventMaskPointerMotion        = 64
	EventMaskPointerMotionHint    = 128
	EventMaskButton1Motion        = 256
	EventMaskButton2Motion        = 512
	EventMaskButton3Motion        = 1024
	EventMaskButton4Motion        = 2048
	EventMaskButton5Motion        = 4096
	EventMaskButtonMotion         = 8192
	EventMaskKeymapState          = 16384
	EventMaskExposure             = 32768
	EventMaskVisibilityChange     = 65536
	EventMaskStructureNotify      = 131072
	EventMaskResizeRedirect       = 262144
	EventMaskSubstructureNotify   = 524288
	EventMaskSubstructureRedirect = 1048576
	EventMaskFocusChange          = 2097152
	EventMaskPropertyChange       = 4194304
	EventMaskColorMapChange       = 8388608
	EventMaskOwnerGrabButton      = 16777216
)

const (
	BackingStoreNotUseful  = 0
	BackingStoreWhenMapped = 1
	BackingStoreAlways     = 2
)

type ScreenInfo struct {
	Root                Id
	DefaultColormap     Id
	WhitePixel          uint32
	BlackPixel          uint32
	CurrentInputMasks   uint32
	WidthInPixels       uint16
	HeightInPixels      uint16
	WidthInMillimeters  uint16
	HeightInMillimeters uint16
	MinInstalledMaps    uint16
	MaxInstalledMaps    uint16
	RootVisual          Id
	BackingStores       byte
	SaveUnders          byte
	RootDepth           byte
	AllowedDepthsLen    byte
	AllowedDepths       []DepthInfo
}

func getScreenInfo(b []byte, v *ScreenInfo) int {
	v.Root = Id(get32(b[0:]))
	v.DefaultColormap = Id(get32(b[4:]))
	v.WhitePixel = get32(b[8:])
	v.BlackPixel = get32(b[12:])
	v.CurrentInputMasks = get32(b[16:])
	v.WidthInPixels = get16(b[20:])
	v.HeightInPixels = get16(b[22:])
	v.WidthInMillimeters = get16(b[24:])
	v.HeightInMillimeters = get16(b[26:])
	v.MinInstalledMaps = get16(b[28:])
	v.MaxInstalledMaps = get16(b[30:])
	v.RootVisual = Id(get32(b[32:]))
	v.BackingStores = b[36]
	v.SaveUnders = b[37]
	v.RootDepth = b[38]
	v.AllowedDepthsLen = b[39]
	offset := 40
	v.AllowedDepths = make([]DepthInfo, int(v.AllowedDepthsLen))
	for i := 0; i < int(v.AllowedDepthsLen); i++ {
		offset += getDepthInfo(b[offset:], &v.AllowedDepths[i])
	}
	return offset
}

const (
	ImageOrderLSBFirst = 0
	ImageOrderMSBFirst = 1
)

type SetupInfo struct {
	Status                   byte
	ProtocolMajorVersion     uint16
	ProtocolMinorVersion     uint16
	Length                   uint16
	ReleaseNumber            uint32
	ResourceIdBase           uint32
	ResourceIdMask           uint32
	MotionBufferSize         uint32
	VendorLen                uint16
	MaximumRequestLength     uint16
	RootsLen                 byte
	PixmapFormatsLen         byte
	ImageByteOrder           byte
	BitmapFormatBitOrder     byte
	BitmapFormatScanlineUnit byte
	BitmapFormatScanlinePad  byte
	MinKeycode               byte
	MaxKeycode               byte
	Vendor                   []byte
	PixmapFormats            []Format
	Roots                    []ScreenInfo
}

func getSetupInfo(b []byte, v *SetupInfo) int {
	v.Status = b[0]
	v.ProtocolMajorVersion = get16(b[2:])
	v.ProtocolMinorVersion = get16(b[4:])
	v.Length = get16(b[6:])
	v.ReleaseNumber = get32(b[8:])
	v.ResourceIdBase = get32(b[12:])
	v.ResourceIdMask = get32(b[16:])
	v.MotionBufferSize = get32(b[20:])
	v.VendorLen = get16(b[24:])
	v.MaximumRequestLength = get16(b[26:])
	v.RootsLen = b[28]
	v.PixmapFormatsLen = b[29]
	v.ImageByteOrder = b[30]
	v.BitmapFormatBitOrder = b[31]
	v.BitmapFormatScanlineUnit = b[32]
	v.BitmapFormatScanlinePad = b[33]
	v.MinKeycode = b[34]
	v.MaxKeycode = b[35]
	offset := 40
	v.Vendor = make([]byte, int(v.VendorLen))
	copy(v.Vendor[0:len(v.Vendor)], b[offset:])
	offset += len(v.Vendor) * 1
	offset = pad(offset)
	v.PixmapFormats = make([]Format, int(v.PixmapFormatsLen))
	for i := 0; i < int(v.PixmapFormatsLen); i++ {
		offset += getFormat(b[offset:], &v.PixmapFormats[i])
	}
	offset = pad(offset)
	v.Roots = make([]ScreenInfo, int(v.RootsLen))
	for i := 0; i < int(v.RootsLen); i++ {
		offset += getScreenInfo(b[offset:], &v.Roots[i])
	}
	return offset
}

const (
	ModMaskShift   = 1
	ModMaskLock    = 2
	ModMaskControl = 4
	ModMask1       = 8
	ModMask2       = 16
	ModMask3       = 32
	ModMask4       = 64
	ModMask5       = 128
	ModMaskAny     = 32768
)

const (
	KeyButMaskShift   = 1
	KeyButMaskLock    = 2
	KeyButMaskControl = 4
	KeyButMaskMod1    = 8
	KeyButMaskMod2    = 16
	KeyButMaskMod3    = 32
	KeyButMaskMod4    = 64
	KeyButMaskMod5    = 128
	KeyButMaskButton1 = 256
	KeyButMaskButton2 = 512
	KeyButMaskButton3 = 1024
	KeyButMaskButton4 = 2048
	KeyButMaskButton5 = 4096
)

const (
	WindowNone = 0
)

const KeyPress = 2

type KeyPressEvent struct {
	Detail     byte
	Time       Timestamp
	Root       Id
	Event      Id
	Child      Id
	RootX      int16
	RootY      int16
	EventX     int16
	EventY     int16
	State      uint16
	SameScreen byte
}

func getKeyPressEvent(b []byte) KeyPressEvent {
	var v KeyPressEvent
	v.Detail = b[1]
	v.Time = Timestamp(get32(b[4:]))
	v.Root = Id(get32(b[8:]))
	v.Event = Id(get32(b[12:]))
	v.Child = Id(get32(b[16:]))
	v.RootX = int16(get16(b[20:]))
	v.RootY = int16(get16(b[22:]))
	v.EventX = int16(get16(b[24:]))
	v.EventY = int16(get16(b[26:]))
	v.State = get16(b[28:])
	v.SameScreen = b[30]
	return v
}

const KeyRelease = 3

type KeyReleaseEvent KeyPressEvent

func getKeyReleaseEvent(b []byte) KeyReleaseEvent {
	return (KeyReleaseEvent)(getKeyPressEvent(b))
}

const (
	ButtonMask1   = 256
	ButtonMask2   = 512
	ButtonMask3   = 1024
	ButtonMask4   = 2048
	ButtonMask5   = 4096
	ButtonMaskAny = 32768
)

const ButtonPress = 4

type ButtonPressEvent struct {
	Detail     byte
	Time       Timestamp
	Root       Id
	Event      Id
	Child      Id
	RootX      int16
	RootY      int16
	EventX     int16
	EventY     int16
	State      uint16
	SameScreen byte
}

func getButtonPressEvent(b []byte) ButtonPressEvent {
	var v ButtonPressEvent
	v.Detail = b[1]
	v.Time = Timestamp(get32(b[4:]))
	v.Root = Id(get32(b[8:]))
	v.Event = Id(get32(b[12:]))
	v.Child = Id(get32(b[16:]))
	v.RootX = int16(get16(b[20:]))
	v.RootY = int16(get16(b[22:]))
	v.EventX = int16(get16(b[24:]))
	v.EventY = int16(get16(b[26:]))
	v.State = get16(b[28:])
	v.SameScreen = b[30]
	return v
}

const ButtonRelease = 5

type ButtonReleaseEvent ButtonPressEvent

func getButtonReleaseEvent(b []byte) ButtonReleaseEvent {
	return (ButtonReleaseEvent)(getButtonPressEvent(b))
}

const (
	MotionNormal = 0
	MotionHint   = 1
)

const MotionNotify = 6

type MotionNotifyEvent struct {
	Detail     byte
	Time       Timestamp
	Root       Id
	Event      Id
	Child      Id
	RootX      int16
	RootY      int16
	EventX     int16
	EventY     int16
	State      uint16
	SameScreen byte
}

func getMotionNotifyEvent(b []byte) MotionNotifyEvent {
	var v MotionNotifyEvent
	v.Detail = b[1]
	v.Time = Timestamp(get32(b[4:]))
	v.Root = Id(get32(b[8:]))
	v.Event = Id(get32(b[12:]))
	v.Child = Id(get32(b[16:]))
	v.RootX = int16(get16(b[20:]))
	v.RootY = int16(get16(b[22:]))
	v.EventX = int16(get16(b[24:]))
	v.EventY = int16(get16(b[26:]))
	v.State = get16(b[28:])
	v.SameScreen = b[30]
	return v
}

const (
	NotifyDetailAncestor         = 0
	NotifyDetailVirtual          = 1
	NotifyDetailInferior         = 2
	NotifyDetailNonlinear        = 3
	NotifyDetailNonlinearVirtual = 4
	NotifyDetailPointer          = 5
	NotifyDetailPointerRoot      = 6
	NotifyDetailNone             = 7
)

const (
	NotifyModeNormal       = 0
	NotifyModeGrab         = 1
	NotifyModeUngrab       = 2
	NotifyModeWhileGrabbed = 3
)

const EnterNotify = 7

type EnterNotifyEvent struct {
	Detail          byte
	Time            Timestamp
	Root            Id
	Event           Id
	Child           Id
	RootX           int16
	RootY           int16
	EventX          int16
	EventY          int16
	State           uint16
	Mode            byte
	SameScreenFocus byte
}

func getEnterNotifyEvent(b []byte) EnterNotifyEvent {
	var v EnterNotifyEvent
	v.Detail = b[1]
	v.Time = Timestamp(get32(b[4:]))
	v.Root = Id(get32(b[8:]))
	v.Event = Id(get32(b[12:]))
	v.Child = Id(get32(b[16:]))
	v.RootX = int16(get16(b[20:]))
	v.RootY = int16(get16(b[22:]))
	v.EventX = int16(get16(b[24:]))
	v.EventY = int16(get16(b[26:]))
	v.State = get16(b[28:])
	v.Mode = b[30]
	v.SameScreenFocus = b[31]
	return v
}

const LeaveNotify = 8

type LeaveNotifyEvent EnterNotifyEvent

func getLeaveNotifyEvent(b []byte) LeaveNotifyEvent {
	return (LeaveNotifyEvent)(getEnterNotifyEvent(b))
}

const FocusIn = 9

type FocusInEvent struct {
	Detail byte
	Event  Id
	Mode   byte
}

func getFocusInEvent(b []byte) FocusInEvent {
	var v FocusInEvent
	v.Detail = b[1]
	v.Event = Id(get32(b[4:]))
	v.Mode = b[8]
	return v
}

const FocusOut = 10

type FocusOutEvent FocusInEvent

func getFocusOutEvent(b []byte) FocusOutEvent { return (FocusOutEvent)(getFocusInEvent(b)) }

const KeymapNotify = 11

type KeymapNotifyEvent struct {
	Keys [31]byte
}

func getKeymapNotifyEvent(b []byte) KeymapNotifyEvent {
	var v KeymapNotifyEvent
	copy(v.Keys[0:31], b[1:])
	return v
}

const Expose = 12

type ExposeEvent struct {
	Window Id
	X      uint16
	Y      uint16
	Width  uint16
	Height uint16
	Count  uint16
}

func getExposeEvent(b []byte) ExposeEvent {
	var v ExposeEvent
	v.Window = Id(get32(b[4:]))
	v.X = get16(b[8:])
	v.Y = get16(b[10:])
	v.Width = get16(b[12:])
	v.Height = get16(b[14:])
	v.Count = get16(b[16:])
	return v
}

const GraphicsExposure = 13

type GraphicsExposureEvent struct {
	Drawable    Id
	X           uint16
	Y           uint16
	Width       uint16
	Height      uint16
	MinorOpcode uint16
	Count       uint16
	MajorOpcode byte
}

func getGraphicsExposureEvent(b []byte) GraphicsExposureEvent {
	var v GraphicsExposureEvent
	v.Drawable = Id(get32(b[4:]))
	v.X = get16(b[8:])
	v.Y = get16(b[10:])
	v.Width = get16(b[12:])
	v.Height = get16(b[14:])
	v.MinorOpcode = get16(b[16:])
	v.Count = get16(b[18:])
	v.MajorOpcode = b[20]
	return v
}

const NoExposure = 14

type NoExposureEvent struct {
	Drawable    Id
	MinorOpcode uint16
	MajorOpcode byte
}

func getNoExposureEvent(b []byte) NoExposureEvent {
	var v NoExposureEvent
	v.Drawable = Id(get32(b[4:]))
	v.MinorOpcode = get16(b[8:])
	v.MajorOpcode = b[10]
	return v
}

const (
	VisibilityUnobscured        = 0
	VisibilityPartiallyObscured = 1
	VisibilityFullyObscured     = 2
)

const VisibilityNotify = 15

type VisibilityNotifyEvent struct {
	Window Id
	State  byte
}

func getVisibilityNotifyEvent(b []byte) VisibilityNotifyEvent {
	var v VisibilityNotifyEvent
	v.Window = Id(get32(b[4:]))
	v.State = b[8]
	return v
}

const CreateNotify = 16

type CreateNotifyEvent struct {
	Parent           Id
	Window           Id
	X                int16
	Y                int16
	Width            uint16
	Height           uint16
	BorderWidth      uint16
	OverrideRedirect byte
}

func getCreateNotifyEvent(b []byte) CreateNotifyEvent {
	var v CreateNotifyEvent
	v.Parent = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.X = int16(get16(b[12:]))
	v.Y = int16(get16(b[14:]))
	v.Width = get16(b[16:])
	v.Height = get16(b[18:])
	v.BorderWidth = get16(b[20:])
	v.OverrideRedirect = b[22]
	return v
}

const DestroyNotify = 17

type DestroyNotifyEvent struct {
	Event  Id
	Window Id
}

func getDestroyNotifyEvent(b []byte) DestroyNotifyEvent {
	var v DestroyNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	return v
}

const UnmapNotify = 18

type UnmapNotifyEvent struct {
	Event         Id
	Window        Id
	FromConfigure byte
}

func getUnmapNotifyEvent(b []byte) UnmapNotifyEvent {
	var v UnmapNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.FromConfigure = b[12]
	return v
}

const MapNotify = 19

type MapNotifyEvent struct {
	Event            Id
	Window           Id
	OverrideRedirect byte
}

func getMapNotifyEvent(b []byte) MapNotifyEvent {
	var v MapNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.OverrideRedirect = b[12]
	return v
}

const MapRequest = 20

type MapRequestEvent struct {
	Parent Id
	Window Id
}

func getMapRequestEvent(b []byte) MapRequestEvent {
	var v MapRequestEvent
	v.Parent = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	return v
}

const ReparentNotify = 21

type ReparentNotifyEvent struct {
	Event            Id
	Window           Id
	Parent           Id
	X                int16
	Y                int16
	OverrideRedirect byte
}

func getReparentNotifyEvent(b []byte) ReparentNotifyEvent {
	var v ReparentNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.Parent = Id(get32(b[12:]))
	v.X = int16(get16(b[16:]))
	v.Y = int16(get16(b[18:]))
	v.OverrideRedirect = b[20]
	return v
}

const ConfigureNotify = 22

type ConfigureNotifyEvent struct {
	Event            Id
	Window           Id
	AboveSibling     Id
	X                int16
	Y                int16
	Width            uint16
	Height           uint16
	BorderWidth      uint16
	OverrideRedirect byte
}

func getConfigureNotifyEvent(b []byte) ConfigureNotifyEvent {
	var v ConfigureNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.AboveSibling = Id(get32(b[12:]))
	v.X = int16(get16(b[16:]))
	v.Y = int16(get16(b[18:]))
	v.Width = get16(b[20:])
	v.Height = get16(b[22:])
	v.BorderWidth = get16(b[24:])
	v.OverrideRedirect = b[26]
	return v
}

const ConfigureRequest = 23

type ConfigureRequestEvent struct {
	StackMode   byte
	Parent      Id
	Window      Id
	Sibling     Id
	X           int16
	Y           int16
	Width       uint16
	Height      uint16
	BorderWidth uint16
	ValueMask   uint16
}

func getConfigureRequestEvent(b []byte) ConfigureRequestEvent {
	var v ConfigureRequestEvent
	v.StackMode = b[1]
	v.Parent = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.Sibling = Id(get32(b[12:]))
	v.X = int16(get16(b[16:]))
	v.Y = int16(get16(b[18:]))
	v.Width = get16(b[20:])
	v.Height = get16(b[22:])
	v.BorderWidth = get16(b[24:])
	v.ValueMask = get16(b[26:])
	return v
}

const GravityNotify = 24

type GravityNotifyEvent struct {
	Event  Id
	Window Id
	X      int16
	Y      int16
}

func getGravityNotifyEvent(b []byte) GravityNotifyEvent {
	var v GravityNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.X = int16(get16(b[12:]))
	v.Y = int16(get16(b[14:]))
	return v
}

const ResizeRequest = 25

type ResizeRequestEvent struct {
	Window Id
	Width  uint16
	Height uint16
}

func getResizeRequestEvent(b []byte) ResizeRequestEvent {
	var v ResizeRequestEvent
	v.Window = Id(get32(b[4:]))
	v.Width = get16(b[8:])
	v.Height = get16(b[10:])
	return v
}

const (
	PlaceOnTop    = 0
	PlaceOnBottom = 1
)

const CirculateNotify = 26

type CirculateNotifyEvent struct {
	Event  Id
	Window Id
	Place  byte
}

func getCirculateNotifyEvent(b []byte) CirculateNotifyEvent {
	var v CirculateNotifyEvent
	v.Event = Id(get32(b[4:]))
	v.Window = Id(get32(b[8:]))
	v.Place = b[16]
	return v
}

const CirculateRequest = 27

type CirculateRequestEvent CirculateNotifyEvent

func getCirculateRequestEvent(b []byte) CirculateRequestEvent {
	return (CirculateRequestEvent)(getCirculateNotifyEvent(b))
}

const (
	PropertyNewValue = 0
	PropertyDelete   = 1
)

const PropertyNotify = 28

type PropertyNotifyEvent struct {
	Window Id
	Atom   Id
	Time   Timestamp
	State  byte
}

func getPropertyNotifyEvent(b []byte) PropertyNotifyEvent {
	var v PropertyNotifyEvent
	v.Window = Id(get32(b[4:]))
	v.Atom = Id(get32(b[8:]))
	v.Time = Timestamp(get32(b[12:]))
	v.State = b[16]
	return v
}

const SelectionClear = 29

type SelectionClearEvent struct {
	Time      Timestamp
	Owner     Id
	Selection Id
}

func getSelectionClearEvent(b []byte) SelectionClearEvent {
	var v SelectionClearEvent
	v.Time = Timestamp(get32(b[4:]))
	v.Owner = Id(get32(b[8:]))
	v.Selection = Id(get32(b[12:]))
	return v
}

const (
	TimeCurrentTime = 0
)

const (
	AtomNone               = 0
	AtomAny                = 0
	AtomPrimary            = 1
	AtomSecondary          = 2
	AtomArc                = 3
	AtomAtom               = 4
	AtomBitmap             = 5
	AtomCardinal           = 6
	AtomColormap           = 7
	AtomCursor             = 8
	AtomCutBuffer0         = 9
	AtomCutBuffer1         = 10
	AtomCutBuffer2         = 11
	AtomCutBuffer3         = 12
	AtomCutBuffer4         = 13
	AtomCutBuffer5         = 14
	AtomCutBuffer6         = 15
	AtomCutBuffer7         = 16
	AtomDrawable           = 17
	AtomFont               = 18
	AtomInteger            = 19
	AtomPixmap             = 20
	AtomPoint              = 21
	AtomRectangle          = 22
	AtomResourceManager    = 23
	AtomRgbColorMap        = 24
	AtomRgbBestMap         = 25
	AtomRgbBlueMap         = 26
	AtomRgbDefaultMap      = 27
	AtomRgbGrayMap         = 28
	AtomRgbGreenMap        = 29
	AtomRgbRedMap          = 30
	AtomString             = 31
	AtomVisualid           = 32
	AtomWindow             = 33
	AtomWmCommand          = 34
	AtomWmHints            = 35
	AtomWmClientMachine    = 36
	AtomWmIconName         = 37
	AtomWmIconSize         = 38
	AtomWmName             = 39
	AtomWmNormalHints      = 40
	AtomWmSizeHints        = 41
	AtomWmZoomHints        = 42
	AtomMinSpace           = 43
	AtomNormSpace          = 44
	AtomMaxSpace           = 45
	AtomEndSpace           = 46
	AtomSuperscriptX       = 47
	AtomSuperscriptY       = 48
	AtomSubscriptX         = 49
	AtomSubscriptY         = 50
	AtomUnderlinePosition  = 51
	AtomUnderlineThickness = 52
	AtomStrikeoutAscent    = 53
	AtomStrikeoutDescent   = 54
	AtomItalicAngle        = 55
	AtomXHeight            = 56
	AtomQuadWidth          = 57
	AtomWeight             = 58
	AtomPointSize          = 59
	AtomResolution         = 60
	AtomCopyright          = 61
	AtomNotice             = 62
	AtomFontName           = 63
	AtomFamilyName         = 64
	AtomFullName           = 65
	AtomCapHeight          = 66
	AtomWmClass            = 67
	AtomWmTransientFor     = 68
)

const SelectionRequest = 30

type SelectionRequestEvent struct {
	Time      Timestamp
	Owner     Id
	Requestor Id
	Selection Id
	Target    Id
	Property  Id
}

func getSelectionRequestEvent(b []byte) SelectionRequestEvent {
	var v SelectionRequestEvent
	v.Time = Timestamp(get32(b[4:]))
	v.Owner = Id(get32(b[8:]))
	v.Requestor = Id(get32(b[12:]))
	v.Selection = Id(get32(b[16:]))
	v.Target = Id(get32(b[20:]))
	v.Property = Id(get32(b[24:]))
	return v
}

const SelectionNotify = 31

type SelectionNotifyEvent struct {
	Time      Timestamp
	Requestor Id
	Selection Id
	Target    Id
	Property  Id
}

func getSelectionNotifyEvent(b []byte) SelectionNotifyEvent {
	var v SelectionNotifyEvent
	v.Time = Timestamp(get32(b[4:]))
	v.Requestor = Id(get32(b[8:]))
	v.Selection = Id(get32(b[12:]))
	v.Target = Id(get32(b[16:]))
	v.Property = Id(get32(b[20:]))
	return v
}

const (
	ColormapStateUninstalled = 0
	ColormapStateInstalled   = 1
)

const (
	ColormapNone = 0
)

const ColormapNotify = 32

type ColormapNotifyEvent struct {
	Window   Id
	Colormap Id
	New      byte
	State    byte
}

func getColormapNotifyEvent(b []byte) ColormapNotifyEvent {
	var v ColormapNotifyEvent
	v.Window = Id(get32(b[4:]))
	v.Colormap = Id(get32(b[8:]))
	v.New = b[12]
	v.State = b[13]
	return v
}

const ClientMessage = 33

type ClientMessageEvent struct {
	Format byte
	Window Id
	Type   Id
	Data   ClientMessageData
}

func getClientMessageEvent(b []byte) ClientMessageEvent {
	var v ClientMessageEvent
	v.Format = b[1]
	v.Window = Id(get32(b[4:]))
	v.Type = Id(get32(b[8:]))
	getClientMessageData(b[12:], &v.Data)
	return v
}

const (
	MappingModifier = 0
	MappingKeyboard = 1
	MappingPointer  = 2
)

const MappingNotify = 34

type MappingNotifyEvent struct {
	Request      byte
	FirstKeycode byte
	Count        byte
}

func getMappingNotifyEvent(b []byte) MappingNotifyEvent {
	var v MappingNotifyEvent
	v.Request = b[4]
	v.FirstKeycode = b[5]
	v.Count = b[6]
	return v
}

const BadRequest = 1

const BadValue = 2

const BadWindow = 3

const BadPixmap = 4

const BadAtom = 5

const BadCursor = 6

const BadFont = 7

const BadMatch = 8

const BadDrawable = 9

const BadAccess = 10

const BadAlloc = 11

const BadColormap = 12

const BadGContext = 13

const BadIDChoice = 14

const BadName = 15

const BadLength = 16

const BadImplementation = 17

const (
	WindowClassCopyFromParent = 0
	WindowClassInputOutput    = 1
	WindowClassInputOnly      = 2
)

const (
	CWBackPixmap       = 1
	CWBackPixel        = 2
	CWBorderPixmap     = 4
	CWBorderPixel      = 8
	CWBitGravity       = 16
	CWWinGravity       = 32
	CWBackingStore     = 64
	CWBackingPlanes    = 128
	CWBackingPixel     = 256
	CWOverrideRedirect = 512
	CWSaveUnder        = 1024
	CWEventMask        = 2048
	CWDontPropagate    = 4096
	CWColormap         = 8192
	CWCursor           = 16384
)

const (
	BackPixmapNone           = 0
	BackPixmapParentRelative = 1
)

const (
	GravityBitForget = 0
	GravityWinUnmap  = 0
	GravityNorthWest = 1
	GravityNorth     = 2
	GravityNorthEast = 3
	GravityWest      = 4
	GravityCenter    = 5
	GravityEast      = 6
	GravitySouthWest = 7
	GravitySouth     = 8
	GravitySouthEast = 9
	GravityStatic    = 10
)

func (c *Conn) CreateWindow(Depth byte, Wid Id, Parent Id, X int16, Y int16, Width uint16, Height uint16, BorderWidth uint16, Class uint16, Visual Id, ValueMask uint32, ValueList []uint32) {
	b := c.scratch[0:32]
	n := 32
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 1
	b[1] = Depth
	put32(b[4:], uint32(Wid))
	put32(b[8:], uint32(Parent))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	put16(b[16:], Width)
	put16(b[18:], Height)
	put16(b[20:], BorderWidth)
	put16(b[22:], Class)
	put32(b[24:], uint32(Visual))
	put32(b[28:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

func (c *Conn) ChangeWindowAttributes(Window Id, ValueMask uint32, ValueList []uint32) {
	b := c.scratch[0:12]
	n := 12
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 2
	put32(b[4:], uint32(Window))
	put32(b[8:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

const (
	MapStateUnmapped   = 0
	MapStateUnviewable = 1
	MapStateViewable   = 2
)

func (c *Conn) GetWindowAttributesRequest(Window Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 3
	put32(b[4:], uint32(Window))
	return c.sendRequest(b)
}

func (c *Conn) GetWindowAttributes(Window Id) (*GetWindowAttributesReply, os.Error) {
	return c.GetWindowAttributesReply(c.GetWindowAttributesRequest(Window))
}

type GetWindowAttributesReply struct {
	BackingStore       byte
	Visual             Id
	Class              uint16
	BitGravity         byte
	WinGravity         byte
	BackingPlanes      uint32
	BackingPixel       uint32
	SaveUnder          byte
	MapIsInstalled     byte
	MapState           byte
	OverrideRedirect   byte
	Colormap           Id
	AllEventMasks      uint32
	YourEventMask      uint32
	DoNotPropagateMask uint16
}

func (c *Conn) GetWindowAttributesReply(cookie Cookie) (*GetWindowAttributesReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetWindowAttributesReply)
	v.BackingStore = b[1]
	v.Visual = Id(get32(b[8:]))
	v.Class = get16(b[12:])
	v.BitGravity = b[14]
	v.WinGravity = b[15]
	v.BackingPlanes = get32(b[16:])
	v.BackingPixel = get32(b[20:])
	v.SaveUnder = b[24]
	v.MapIsInstalled = b[25]
	v.MapState = b[26]
	v.OverrideRedirect = b[27]
	v.Colormap = Id(get32(b[28:]))
	v.AllEventMasks = get32(b[32:])
	v.YourEventMask = get32(b[36:])
	v.DoNotPropagateMask = get16(b[40:])
	return v, nil
}

func (c *Conn) DestroyWindow(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 4
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) DestroySubwindows(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 5
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

const (
	SetModeInsert = 0
	SetModeDelete = 1
)

func (c *Conn) ChangeSaveSet(Mode byte, Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 6
	b[1] = Mode
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) ReparentWindow(Window Id, Parent Id, X int16, Y int16) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 7
	put32(b[4:], uint32(Window))
	put32(b[8:], uint32(Parent))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	c.sendRequest(b)
}

func (c *Conn) MapWindow(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 8
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) MapSubwindows(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 9
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) UnmapWindow(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 10
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) UnmapSubwindows(Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 11
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

const (
	ConfigWindowX           = 1
	ConfigWindowY           = 2
	ConfigWindowWidth       = 4
	ConfigWindowHeight      = 8
	ConfigWindowBorderWidth = 16
	ConfigWindowSibling     = 32
	ConfigWindowStackMode   = 64
)

const (
	StackModeAbove    = 0
	StackModeBelow    = 1
	StackModeTopIf    = 2
	StackModeBottomIf = 3
	StackModeOpposite = 4
)

func (c *Conn) ConfigureWindow(Window Id, ValueMask uint16, ValueList []uint32) {
	b := c.scratch[0:12]
	n := 12
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 12
	put32(b[4:], uint32(Window))
	put16(b[8:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

const (
	CirculateRaiseLowest  = 0
	CirculateLowerHighest = 1
)

func (c *Conn) CirculateWindow(Direction byte, Window Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 13
	b[1] = Direction
	put32(b[4:], uint32(Window))
	c.sendRequest(b)
}

func (c *Conn) GetGeometryRequest(Drawable Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 14
	put32(b[4:], uint32(Drawable))
	return c.sendRequest(b)
}

func (c *Conn) GetGeometry(Drawable Id) (*GetGeometryReply, os.Error) {
	return c.GetGeometryReply(c.GetGeometryRequest(Drawable))
}

type GetGeometryReply struct {
	Depth       byte
	Root        Id
	X           int16
	Y           int16
	Width       uint16
	Height      uint16
	BorderWidth uint16
}

func (c *Conn) GetGeometryReply(cookie Cookie) (*GetGeometryReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetGeometryReply)
	v.Depth = b[1]
	v.Root = Id(get32(b[8:]))
	v.X = int16(get16(b[12:]))
	v.Y = int16(get16(b[14:]))
	v.Width = get16(b[16:])
	v.Height = get16(b[18:])
	v.BorderWidth = get16(b[20:])
	return v, nil
}

func (c *Conn) QueryTreeRequest(Window Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 15
	put32(b[4:], uint32(Window))
	return c.sendRequest(b)
}

func (c *Conn) QueryTree(Window Id) (*QueryTreeReply, os.Error) {
	return c.QueryTreeReply(c.QueryTreeRequest(Window))
}

type QueryTreeReply struct {
	Root        Id
	Parent      Id
	ChildrenLen uint16
	Children    []Id
}

func (c *Conn) QueryTreeReply(cookie Cookie) (*QueryTreeReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryTreeReply)
	v.Root = Id(get32(b[8:]))
	v.Parent = Id(get32(b[12:]))
	v.ChildrenLen = get16(b[16:])
	offset := 32
	v.Children = make([]Id, int(v.ChildrenLen))
	for i := 0; i < len(v.Children); i++ {
		v.Children[i] = Id(get32(b[offset+i*4:]))
	}
	offset += len(v.Children) * 4
	return v, nil
}

func (c *Conn) InternAtomRequest(OnlyIfExists byte, Name string) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 16
	b[1] = OnlyIfExists
	put16(b[4:], uint16(len(Name)))
	cookie := c.sendRequest(b)
	c.sendString(Name)
	return cookie
}

func (c *Conn) InternAtom(OnlyIfExists byte, Name string) (*InternAtomReply, os.Error) {
	return c.InternAtomReply(c.InternAtomRequest(OnlyIfExists, Name))
}

type InternAtomReply struct {
	Atom Id
}

func (c *Conn) InternAtomReply(cookie Cookie) (*InternAtomReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(InternAtomReply)
	v.Atom = Id(get32(b[8:]))
	return v, nil
}

func (c *Conn) GetAtomNameRequest(Atom Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 17
	put32(b[4:], uint32(Atom))
	return c.sendRequest(b)
}

func (c *Conn) GetAtomName(Atom Id) (*GetAtomNameReply, os.Error) {
	return c.GetAtomNameReply(c.GetAtomNameRequest(Atom))
}

type GetAtomNameReply struct {
	NameLen uint16
	Name    []byte
}

func (c *Conn) GetAtomNameReply(cookie Cookie) (*GetAtomNameReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetAtomNameReply)
	v.NameLen = get16(b[8:])
	offset := 32
	v.Name = make([]byte, int(v.NameLen))
	copy(v.Name[0:len(v.Name)], b[offset:])
	offset += len(v.Name) * 1
	return v, nil
}

const (
	PropModeReplace = 0
	PropModePrepend = 1
	PropModeAppend  = 2
)

func (c *Conn) ChangeProperty(Mode byte, Window Id, Property Id, Type Id, Format byte, Data []byte) {
	b := c.scratch[0:24]
	n := 24
	n += pad(((len(Data) * int(Format)) / 8) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 18
	b[1] = Mode
	put32(b[4:], uint32(Window))
	put32(b[8:], uint32(Property))
	put32(b[12:], uint32(Type))
	b[16] = Format
	put32(b[20:], uint32(len(Data)))
	c.sendRequest(b)
	c.sendBytes(Data[0:((len(Data) * int(Format)) / 8)])
}

func (c *Conn) DeleteProperty(Window Id, Property Id) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 19
	put32(b[4:], uint32(Window))
	put32(b[8:], uint32(Property))
	c.sendRequest(b)
}

const (
	GetPropertyTypeAny = 0
)

func (c *Conn) GetPropertyRequest(Delete byte, Window Id, Property Id, Type Id, LongOffset uint32, LongLength uint32) Cookie {
	b := c.scratch[0:24]
	put16(b[2:], 6)
	b[0] = 20
	b[1] = Delete
	put32(b[4:], uint32(Window))
	put32(b[8:], uint32(Property))
	put32(b[12:], uint32(Type))
	put32(b[16:], LongOffset)
	put32(b[20:], LongLength)
	return c.sendRequest(b)
}

func (c *Conn) GetProperty(Delete byte, Window Id, Property Id, Type Id, LongOffset uint32, LongLength uint32) (*GetPropertyReply, os.Error) {
	return c.GetPropertyReply(c.GetPropertyRequest(Delete, Window, Property, Type, LongOffset, LongLength))
}

type GetPropertyReply struct {
	Format     byte
	Type       Id
	BytesAfter uint32
	ValueLen   uint32
	Value      []byte
}

func (c *Conn) GetPropertyReply(cookie Cookie) (*GetPropertyReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetPropertyReply)
	v.Format = b[1]
	v.Type = Id(get32(b[8:]))
	v.BytesAfter = get32(b[12:])
	v.ValueLen = get32(b[16:])
	offset := 32
	v.Value = make([]byte, (int(v.ValueLen) * (int(v.Format) / 8)))
	copy(v.Value[0:len(v.Value)], b[offset:])
	offset += len(v.Value) * 1
	return v, nil
}

func (c *Conn) ListPropertiesRequest(Window Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 21
	put32(b[4:], uint32(Window))
	return c.sendRequest(b)
}

func (c *Conn) ListProperties(Window Id) (*ListPropertiesReply, os.Error) {
	return c.ListPropertiesReply(c.ListPropertiesRequest(Window))
}

type ListPropertiesReply struct {
	AtomsLen uint16
	Atoms    []Id
}

func (c *Conn) ListPropertiesReply(cookie Cookie) (*ListPropertiesReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListPropertiesReply)
	v.AtomsLen = get16(b[8:])
	offset := 32
	v.Atoms = make([]Id, int(v.AtomsLen))
	for i := 0; i < len(v.Atoms); i++ {
		v.Atoms[i] = Id(get32(b[offset+i*4:]))
	}
	offset += len(v.Atoms) * 4
	return v, nil
}

func (c *Conn) SetSelectionOwner(Owner Id, Selection Id, Time Timestamp) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 22
	put32(b[4:], uint32(Owner))
	put32(b[8:], uint32(Selection))
	put32(b[12:], uint32(Time))
	c.sendRequest(b)
}

func (c *Conn) GetSelectionOwnerRequest(Selection Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 23
	put32(b[4:], uint32(Selection))
	return c.sendRequest(b)
}

func (c *Conn) GetSelectionOwner(Selection Id) (*GetSelectionOwnerReply, os.Error) {
	return c.GetSelectionOwnerReply(c.GetSelectionOwnerRequest(Selection))
}

type GetSelectionOwnerReply struct {
	Owner Id
}

func (c *Conn) GetSelectionOwnerReply(cookie Cookie) (*GetSelectionOwnerReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetSelectionOwnerReply)
	v.Owner = Id(get32(b[8:]))
	return v, nil
}

func (c *Conn) ConvertSelection(Requestor Id, Selection Id, Target Id, Property Id, Time Timestamp) {
	b := c.scratch[0:24]
	put16(b[2:], 6)
	b[0] = 24
	put32(b[4:], uint32(Requestor))
	put32(b[8:], uint32(Selection))
	put32(b[12:], uint32(Target))
	put32(b[16:], uint32(Property))
	put32(b[20:], uint32(Time))
	c.sendRequest(b)
}

const (
	SendEventDestPointerWindow = 0
	SendEventDestItemFocus     = 1
)

func (c *Conn) SendEvent(Propagate byte, Destination Id, EventMask uint32, Event []byte) {
	b := make([]byte, 44)
	put16(b[2:], 11)
	b[0] = 25
	b[1] = Propagate
	put32(b[4:], uint32(Destination))
	put32(b[8:], EventMask)
	copy(b[12:44], Event)
	c.sendRequest(b)
}

const (
	GrabModeSync  = 0
	GrabModeAsync = 1
)

const (
	GrabStatusSuccess        = 0
	GrabStatusAlreadyGrabbed = 1
	GrabStatusInvalidTime    = 2
	GrabStatusNotViewable    = 3
	GrabStatusFrozen         = 4
)

const (
	CursorNone = 0
)

func (c *Conn) GrabPointerRequest(OwnerEvents byte, GrabWindow Id, EventMask uint16, PointerMode byte, KeyboardMode byte, ConfineTo Id, Cursor Id, Time Timestamp) Cookie {
	b := c.scratch[0:24]
	put16(b[2:], 6)
	b[0] = 26
	b[1] = OwnerEvents
	put32(b[4:], uint32(GrabWindow))
	put16(b[8:], EventMask)
	b[10] = PointerMode
	b[11] = KeyboardMode
	put32(b[12:], uint32(ConfineTo))
	put32(b[16:], uint32(Cursor))
	put32(b[20:], uint32(Time))
	return c.sendRequest(b)
}

func (c *Conn) GrabPointer(OwnerEvents byte, GrabWindow Id, EventMask uint16, PointerMode byte, KeyboardMode byte, ConfineTo Id, Cursor Id, Time Timestamp) (*GrabPointerReply, os.Error) {
	return c.GrabPointerReply(c.GrabPointerRequest(OwnerEvents, GrabWindow, EventMask, PointerMode, KeyboardMode, ConfineTo, Cursor, Time))
}

type GrabPointerReply struct {
	Status byte
}

func (c *Conn) GrabPointerReply(cookie Cookie) (*GrabPointerReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GrabPointerReply)
	v.Status = b[1]
	return v, nil
}

func (c *Conn) UngrabPointer(Time Timestamp) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 27
	put32(b[4:], uint32(Time))
	c.sendRequest(b)
}

const (
	ButtonIndexAny = 0
	ButtonIndex1   = 1
	ButtonIndex2   = 2
	ButtonIndex3   = 3
	ButtonIndex4   = 4
	ButtonIndex5   = 5
)

func (c *Conn) GrabButton(OwnerEvents byte, GrabWindow Id, EventMask uint16, PointerMode byte, KeyboardMode byte, ConfineTo Id, Cursor Id, Button byte, Modifiers uint16) {
	b := c.scratch[0:24]
	put16(b[2:], 6)
	b[0] = 28
	b[1] = OwnerEvents
	put32(b[4:], uint32(GrabWindow))
	put16(b[8:], EventMask)
	b[10] = PointerMode
	b[11] = KeyboardMode
	put32(b[12:], uint32(ConfineTo))
	put32(b[16:], uint32(Cursor))
	b[20] = Button
	put16(b[22:], Modifiers)
	c.sendRequest(b)
}

func (c *Conn) UngrabButton(Button byte, GrabWindow Id, Modifiers uint16) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 29
	b[1] = Button
	put32(b[4:], uint32(GrabWindow))
	put16(b[8:], Modifiers)
	c.sendRequest(b)
}

func (c *Conn) ChangeActivePointerGrab(Cursor Id, Time Timestamp, EventMask uint16) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 30
	put32(b[4:], uint32(Cursor))
	put32(b[8:], uint32(Time))
	put16(b[12:], EventMask)
	c.sendRequest(b)
}

func (c *Conn) GrabKeyboardRequest(OwnerEvents byte, GrabWindow Id, Time Timestamp, PointerMode byte, KeyboardMode byte) Cookie {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 31
	b[1] = OwnerEvents
	put32(b[4:], uint32(GrabWindow))
	put32(b[8:], uint32(Time))
	b[12] = PointerMode
	b[13] = KeyboardMode
	return c.sendRequest(b)
}

func (c *Conn) GrabKeyboard(OwnerEvents byte, GrabWindow Id, Time Timestamp, PointerMode byte, KeyboardMode byte) (*GrabKeyboardReply, os.Error) {
	return c.GrabKeyboardReply(c.GrabKeyboardRequest(OwnerEvents, GrabWindow, Time, PointerMode, KeyboardMode))
}

type GrabKeyboardReply struct {
	Status byte
}

func (c *Conn) GrabKeyboardReply(cookie Cookie) (*GrabKeyboardReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GrabKeyboardReply)
	v.Status = b[1]
	return v, nil
}

func (c *Conn) UngrabKeyboard(Time Timestamp) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 32
	put32(b[4:], uint32(Time))
	c.sendRequest(b)
}

const (
	GrabAny = 0
)

func (c *Conn) GrabKey(OwnerEvents byte, GrabWindow Id, Modifiers uint16, Key byte, PointerMode byte, KeyboardMode byte) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 33
	b[1] = OwnerEvents
	put32(b[4:], uint32(GrabWindow))
	put16(b[8:], Modifiers)
	b[10] = Key
	b[11] = PointerMode
	b[12] = KeyboardMode
	c.sendRequest(b)
}

func (c *Conn) UngrabKey(Key byte, GrabWindow Id, Modifiers uint16) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 34
	b[1] = Key
	put32(b[4:], uint32(GrabWindow))
	put16(b[8:], Modifiers)
	c.sendRequest(b)
}

const (
	AllowAsyncPointer   = 0
	AllowSyncPointer    = 1
	AllowReplayPointer  = 2
	AllowAsyncKeyboard  = 3
	AllowSyncKeyboard   = 4
	AllowReplayKeyboard = 5
	AllowAsyncBoth      = 6
	AllowSyncBoth       = 7
)

func (c *Conn) AllowEvents(Mode byte, Time Timestamp) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 35
	b[1] = Mode
	put32(b[4:], uint32(Time))
	c.sendRequest(b)
}

func (c *Conn) GrabServer() {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 36
	c.sendRequest(b)
}

func (c *Conn) UngrabServer() {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 37
	c.sendRequest(b)
}

func (c *Conn) QueryPointerRequest(Window Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 38
	put32(b[4:], uint32(Window))
	return c.sendRequest(b)
}

func (c *Conn) QueryPointer(Window Id) (*QueryPointerReply, os.Error) {
	return c.QueryPointerReply(c.QueryPointerRequest(Window))
}

type QueryPointerReply struct {
	SameScreen byte
	Root       Id
	Child      Id
	RootX      int16
	RootY      int16
	WinX       int16
	WinY       int16
	Mask       uint16
}

func (c *Conn) QueryPointerReply(cookie Cookie) (*QueryPointerReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryPointerReply)
	v.SameScreen = b[1]
	v.Root = Id(get32(b[8:]))
	v.Child = Id(get32(b[12:]))
	v.RootX = int16(get16(b[16:]))
	v.RootY = int16(get16(b[18:]))
	v.WinX = int16(get16(b[20:]))
	v.WinY = int16(get16(b[22:]))
	v.Mask = get16(b[24:])
	return v, nil
}

type Timecoord struct {
	Time Timestamp
	X    int16
	Y    int16
}

func getTimecoord(b []byte, v *Timecoord) int {
	v.Time = Timestamp(get32(b[0:]))
	v.X = int16(get16(b[4:]))
	v.Y = int16(get16(b[6:]))
	return 8
}

func (c *Conn) sendTimecoordList(list []Timecoord, count int) {
	b0 := make([]byte, 8*count)
	for k := 0; k < count; k++ {
		b := b0[k*8:]
		put32(b[0:], uint32(list[k].Time))
		put16(b[4:], uint16(list[k].X))
		put16(b[6:], uint16(list[k].Y))
	}
	c.sendBytes(b0)
}

func (c *Conn) GetMotionEventsRequest(Window Id, Start Timestamp, Stop Timestamp) Cookie {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 39
	put32(b[4:], uint32(Window))
	put32(b[8:], uint32(Start))
	put32(b[12:], uint32(Stop))
	return c.sendRequest(b)
}

func (c *Conn) GetMotionEvents(Window Id, Start Timestamp, Stop Timestamp) (*GetMotionEventsReply, os.Error) {
	return c.GetMotionEventsReply(c.GetMotionEventsRequest(Window, Start, Stop))
}

type GetMotionEventsReply struct {
	EventsLen uint32
	Events    []Timecoord
}

func (c *Conn) GetMotionEventsReply(cookie Cookie) (*GetMotionEventsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetMotionEventsReply)
	v.EventsLen = get32(b[8:])
	offset := 32
	v.Events = make([]Timecoord, int(v.EventsLen))
	for i := 0; i < int(v.EventsLen); i++ {
		offset += getTimecoord(b[offset:], &v.Events[i])
	}
	return v, nil
}

func (c *Conn) TranslateCoordinatesRequest(SrcWindow Id, DstWindow Id, SrcX int16, SrcY int16) Cookie {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 40
	put32(b[4:], uint32(SrcWindow))
	put32(b[8:], uint32(DstWindow))
	put16(b[12:], uint16(SrcX))
	put16(b[14:], uint16(SrcY))
	return c.sendRequest(b)
}

func (c *Conn) TranslateCoordinates(SrcWindow Id, DstWindow Id, SrcX int16, SrcY int16) (*TranslateCoordinatesReply, os.Error) {
	return c.TranslateCoordinatesReply(c.TranslateCoordinatesRequest(SrcWindow, DstWindow, SrcX, SrcY))
}

type TranslateCoordinatesReply struct {
	SameScreen byte
	Child      Id
	DstX       uint16
	DstY       uint16
}

func (c *Conn) TranslateCoordinatesReply(cookie Cookie) (*TranslateCoordinatesReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(TranslateCoordinatesReply)
	v.SameScreen = b[1]
	v.Child = Id(get32(b[8:]))
	v.DstX = get16(b[12:])
	v.DstY = get16(b[14:])
	return v, nil
}

func (c *Conn) WarpPointer(SrcWindow Id, DstWindow Id, SrcX int16, SrcY int16, SrcWidth uint16, SrcHeight uint16, DstX int16, DstY int16) {
	b := c.scratch[0:24]
	put16(b[2:], 6)
	b[0] = 41
	put32(b[4:], uint32(SrcWindow))
	put32(b[8:], uint32(DstWindow))
	put16(b[12:], uint16(SrcX))
	put16(b[14:], uint16(SrcY))
	put16(b[16:], SrcWidth)
	put16(b[18:], SrcHeight)
	put16(b[20:], uint16(DstX))
	put16(b[22:], uint16(DstY))
	c.sendRequest(b)
}

const (
	InputFocusNone           = 0
	InputFocusPointerRoot    = 1
	InputFocusParent         = 2
	InputFocusFollowKeyboard = 3
)

func (c *Conn) SetInputFocus(RevertTo byte, Focus Id, Time Timestamp) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 42
	b[1] = RevertTo
	put32(b[4:], uint32(Focus))
	put32(b[8:], uint32(Time))
	c.sendRequest(b)
}

func (c *Conn) GetInputFocusRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 43
	return c.sendRequest(b)
}

func (c *Conn) GetInputFocus() (*GetInputFocusReply, os.Error) {
	return c.GetInputFocusReply(c.GetInputFocusRequest())
}

type GetInputFocusReply struct {
	RevertTo byte
	Focus    Id
}

func (c *Conn) GetInputFocusReply(cookie Cookie) (*GetInputFocusReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetInputFocusReply)
	v.RevertTo = b[1]
	v.Focus = Id(get32(b[8:]))
	return v, nil
}

func (c *Conn) QueryKeymapRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 44
	return c.sendRequest(b)
}

func (c *Conn) QueryKeymap() (*QueryKeymapReply, os.Error) {
	return c.QueryKeymapReply(c.QueryKeymapRequest())
}

type QueryKeymapReply struct {
	Keys [32]byte
}

func (c *Conn) QueryKeymapReply(cookie Cookie) (*QueryKeymapReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryKeymapReply)
	copy(v.Keys[0:32], b[8:])
	return v, nil
}

func (c *Conn) OpenFont(Fid Id, Name string) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 45
	put32(b[4:], uint32(Fid))
	put16(b[8:], uint16(len(Name)))
	c.sendRequest(b)
	c.sendString(Name)
}

func (c *Conn) CloseFont(Font Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 46
	put32(b[4:], uint32(Font))
	c.sendRequest(b)
}

const (
	FontDrawLeftToRight = 0
	FontDrawRightToLeft = 1
)

type Fontprop struct {
	Name  Id
	Value uint32
}

func getFontprop(b []byte, v *Fontprop) int {
	v.Name = Id(get32(b[0:]))
	v.Value = get32(b[4:])
	return 8
}

func (c *Conn) sendFontpropList(list []Fontprop, count int) {
	b0 := make([]byte, 8*count)
	for k := 0; k < count; k++ {
		b := b0[k*8:]
		put32(b[0:], uint32(list[k].Name))
		put32(b[4:], list[k].Value)
	}
	c.sendBytes(b0)
}

type Charinfo struct {
	LeftSideBearing  int16
	RightSideBearing int16
	CharacterWidth   int16
	Ascent           int16
	Descent          int16
	Attributes       uint16
}

func getCharinfo(b []byte, v *Charinfo) int {
	v.LeftSideBearing = int16(get16(b[0:]))
	v.RightSideBearing = int16(get16(b[2:]))
	v.CharacterWidth = int16(get16(b[4:]))
	v.Ascent = int16(get16(b[6:]))
	v.Descent = int16(get16(b[8:]))
	v.Attributes = get16(b[10:])
	return 12
}

func (c *Conn) sendCharinfoList(list []Charinfo, count int) {
	b0 := make([]byte, 12*count)
	for k := 0; k < count; k++ {
		b := b0[k*12:]
		put16(b[0:], uint16(list[k].LeftSideBearing))
		put16(b[2:], uint16(list[k].RightSideBearing))
		put16(b[4:], uint16(list[k].CharacterWidth))
		put16(b[6:], uint16(list[k].Ascent))
		put16(b[8:], uint16(list[k].Descent))
		put16(b[10:], list[k].Attributes)
	}
	c.sendBytes(b0)
}

func (c *Conn) QueryFontRequest(Font Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 47
	put32(b[4:], uint32(Font))
	return c.sendRequest(b)
}

func (c *Conn) QueryFont(Font Id) (*QueryFontReply, os.Error) {
	return c.QueryFontReply(c.QueryFontRequest(Font))
}

type QueryFontReply struct {
	MinBounds      Charinfo
	MaxBounds      Charinfo
	MinCharOrByte2 uint16
	MaxCharOrByte2 uint16
	DefaultChar    uint16
	PropertiesLen  uint16
	DrawDirection  byte
	MinByte1       byte
	MaxByte1       byte
	AllCharsExist  byte
	FontAscent     int16
	FontDescent    int16
	CharInfosLen   uint32
	Properties     []Fontprop
	CharInfos      []Charinfo
}

func (c *Conn) QueryFontReply(cookie Cookie) (*QueryFontReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryFontReply)
	getCharinfo(b[8:], &v.MinBounds)
	getCharinfo(b[24:], &v.MaxBounds)
	v.MinCharOrByte2 = get16(b[40:])
	v.MaxCharOrByte2 = get16(b[42:])
	v.DefaultChar = get16(b[44:])
	v.PropertiesLen = get16(b[46:])
	v.DrawDirection = b[48]
	v.MinByte1 = b[49]
	v.MaxByte1 = b[50]
	v.AllCharsExist = b[51]
	v.FontAscent = int16(get16(b[52:]))
	v.FontDescent = int16(get16(b[54:]))
	v.CharInfosLen = get32(b[56:])
	offset := 60
	v.Properties = make([]Fontprop, int(v.PropertiesLen))
	for i := 0; i < int(v.PropertiesLen); i++ {
		offset += getFontprop(b[offset:], &v.Properties[i])
	}
	offset = pad(offset)
	v.CharInfos = make([]Charinfo, int(v.CharInfosLen))
	for i := 0; i < int(v.CharInfosLen); i++ {
		offset += getCharinfo(b[offset:], &v.CharInfos[i])
	}
	return v, nil
}

func (c *Conn) QueryTextExtentsRequest(Font Id, String []Char2b) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(String) * 2)
	put16(b[2:], uint16(n/4))
	b[0] = 48
	b[1] = byte((len(String) & 1))
	put32(b[4:], uint32(Font))
	cookie := c.sendRequest(b)
	c.sendChar2bList(String, len(String))
	return cookie
}

func (c *Conn) QueryTextExtents(Font Id, String []Char2b) (*QueryTextExtentsReply, os.Error) {
	return c.QueryTextExtentsReply(c.QueryTextExtentsRequest(Font, String))
}

type QueryTextExtentsReply struct {
	DrawDirection  byte
	FontAscent     int16
	FontDescent    int16
	OverallAscent  int16
	OverallDescent int16
	OverallWidth   int32
	OverallLeft    int32
	OverallRight   int32
}

func (c *Conn) QueryTextExtentsReply(cookie Cookie) (*QueryTextExtentsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryTextExtentsReply)
	v.DrawDirection = b[1]
	v.FontAscent = int16(get16(b[8:]))
	v.FontDescent = int16(get16(b[10:]))
	v.OverallAscent = int16(get16(b[12:]))
	v.OverallDescent = int16(get16(b[14:]))
	v.OverallWidth = int32(get32(b[16:]))
	v.OverallLeft = int32(get32(b[20:]))
	v.OverallRight = int32(get32(b[24:]))
	return v, nil
}

type Str struct {
	NameLen byte
	Name    []byte
}

func getStr(b []byte, v *Str) int {
	v.NameLen = b[0]
	offset := 1
	v.Name = make([]byte, int(v.NameLen))
	copy(v.Name[0:len(v.Name)], b[offset:])
	offset += len(v.Name) * 1
	return offset
}

// omitting variable length sendStr

func (c *Conn) ListFontsRequest(MaxNames uint16, Pattern []byte) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Pattern) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 49
	put16(b[4:], MaxNames)
	put16(b[6:], uint16(len(Pattern)))
	cookie := c.sendRequest(b)
	c.sendBytes(Pattern[0:len(Pattern)])
	return cookie
}

func (c *Conn) ListFonts(MaxNames uint16, Pattern []byte) (*ListFontsReply, os.Error) {
	return c.ListFontsReply(c.ListFontsRequest(MaxNames, Pattern))
}

type ListFontsReply struct {
	NamesLen uint16
	Names    []Str
}

func (c *Conn) ListFontsReply(cookie Cookie) (*ListFontsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListFontsReply)
	v.NamesLen = get16(b[8:])
	offset := 32
	v.Names = make([]Str, int(v.NamesLen))
	for i := 0; i < int(v.NamesLen); i++ {
		offset += getStr(b[offset:], &v.Names[i])
	}
	return v, nil
}

func (c *Conn) ListFontsWithInfoRequest(MaxNames uint16, Pattern []byte) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Pattern) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 50
	put16(b[4:], MaxNames)
	put16(b[6:], uint16(len(Pattern)))
	cookie := c.sendRequest(b)
	c.sendBytes(Pattern[0:len(Pattern)])
	return cookie
}

func (c *Conn) ListFontsWithInfo(MaxNames uint16, Pattern []byte) (*ListFontsWithInfoReply, os.Error) {
	return c.ListFontsWithInfoReply(c.ListFontsWithInfoRequest(MaxNames, Pattern))
}

type ListFontsWithInfoReply struct {
	NameLen        byte
	MinBounds      Charinfo
	MaxBounds      Charinfo
	MinCharOrByte2 uint16
	MaxCharOrByte2 uint16
	DefaultChar    uint16
	PropertiesLen  uint16
	DrawDirection  byte
	MinByte1       byte
	MaxByte1       byte
	AllCharsExist  byte
	FontAscent     int16
	FontDescent    int16
	RepliesHint    uint32
	Properties     []Fontprop
	Name           []byte
}

func (c *Conn) ListFontsWithInfoReply(cookie Cookie) (*ListFontsWithInfoReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListFontsWithInfoReply)
	v.NameLen = b[1]
	getCharinfo(b[8:], &v.MinBounds)
	getCharinfo(b[24:], &v.MaxBounds)
	v.MinCharOrByte2 = get16(b[40:])
	v.MaxCharOrByte2 = get16(b[42:])
	v.DefaultChar = get16(b[44:])
	v.PropertiesLen = get16(b[46:])
	v.DrawDirection = b[48]
	v.MinByte1 = b[49]
	v.MaxByte1 = b[50]
	v.AllCharsExist = b[51]
	v.FontAscent = int16(get16(b[52:]))
	v.FontDescent = int16(get16(b[54:]))
	v.RepliesHint = get32(b[56:])
	offset := 60
	v.Properties = make([]Fontprop, int(v.PropertiesLen))
	for i := 0; i < int(v.PropertiesLen); i++ {
		offset += getFontprop(b[offset:], &v.Properties[i])
	}
	offset = pad(offset)
	v.Name = make([]byte, int(v.NameLen))
	copy(v.Name[0:len(v.Name)], b[offset:])
	offset += len(v.Name) * 1
	return v, nil
}

func (c *Conn) SetFontPath(FontQty uint16, Path []byte) {
	b := c.scratch[0:6]
	n := 6
	n += pad(len(Path) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 51
	put16(b[4:], FontQty)
	c.sendRequest(b)
	c.sendBytes(Path[0:len(Path)])
}

func (c *Conn) GetFontPathRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 52
	return c.sendRequest(b)
}

func (c *Conn) GetFontPath() (*GetFontPathReply, os.Error) {
	return c.GetFontPathReply(c.GetFontPathRequest())
}

type GetFontPathReply struct {
	PathLen uint16
	Path    []Str
}

func (c *Conn) GetFontPathReply(cookie Cookie) (*GetFontPathReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetFontPathReply)
	v.PathLen = get16(b[8:])
	offset := 32
	v.Path = make([]Str, int(v.PathLen))
	for i := 0; i < int(v.PathLen); i++ {
		offset += getStr(b[offset:], &v.Path[i])
	}
	return v, nil
}

func (c *Conn) CreatePixmap(Depth byte, Pid Id, Drawable Id, Width uint16, Height uint16) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 53
	b[1] = Depth
	put32(b[4:], uint32(Pid))
	put32(b[8:], uint32(Drawable))
	put16(b[12:], Width)
	put16(b[14:], Height)
	c.sendRequest(b)
}

func (c *Conn) FreePixmap(Pixmap Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 54
	put32(b[4:], uint32(Pixmap))
	c.sendRequest(b)
}

const (
	GCFunction           = 1
	GCPlaneMask          = 2
	GCForeground         = 4
	GCBackground         = 8
	GCLineWidth          = 16
	GCLineStyle          = 32
	GCCapStyle           = 64
	GCJoinStyle          = 128
	GCFillStyle          = 256
	GCFillRule           = 512
	GCTile               = 1024
	GCStipple            = 2048
	GCTileStippleOriginX = 4096
	GCTileStippleOriginY = 8192
	GCFont               = 16384
	GCSubwindowMode      = 32768
	GCGraphicsExposures  = 65536
	GCClipOriginX        = 131072
	GCClipOriginY        = 262144
	GCClipMask           = 524288
	GCDashOffset         = 1048576
	GCDashList           = 2097152
	GCArcMode            = 4194304
)

const (
	GXClear        = 0
	GXAnd          = 1
	GXAndReverse   = 2
	GXCopy         = 3
	GXAndInverted  = 4
	GXNoop         = 5
	GXXor          = 6
	GXOr           = 7
	GXNor          = 8
	GXEquiv        = 9
	GXInvert       = 10
	GXOrReverse    = 11
	GXCopyInverted = 12
	GXOrInverted   = 13
	GXNand         = 14
	GXSet          = 15
)

const (
	LineStyleSolid      = 0
	LineStyleOnOffDash  = 1
	LineStyleDoubleDash = 2
)

const (
	CapStyleNotLast    = 0
	CapStyleButt       = 1
	CapStyleRound      = 2
	CapStyleProjecting = 3
)

const (
	JoinStyleMiter = 0
	JoinStyleRound = 1
	JoinStyleBevel = 2
)

const (
	FillStyleSolid          = 0
	FillStyleTiled          = 1
	FillStyleStippled       = 2
	FillStyleOpaqueStippled = 3
)

const (
	FillRuleEvenOdd = 0
	FillRuleWinding = 1
)

const (
	SubwindowModeClipByChildren   = 0
	SubwindowModeIncludeInferiors = 1
)

const (
	ArcModeChord    = 0
	ArcModePieSlice = 1
)

func (c *Conn) CreateGC(Cid Id, Drawable Id, ValueMask uint32, ValueList []uint32) {
	b := c.scratch[0:16]
	n := 16
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 55
	put32(b[4:], uint32(Cid))
	put32(b[8:], uint32(Drawable))
	put32(b[12:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

func (c *Conn) ChangeGC(Gc Id, ValueMask uint32, ValueList []uint32) {
	b := c.scratch[0:12]
	n := 12
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 56
	put32(b[4:], uint32(Gc))
	put32(b[8:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

func (c *Conn) CopyGC(SrcGc Id, DstGc Id, ValueMask uint32) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 57
	put32(b[4:], uint32(SrcGc))
	put32(b[8:], uint32(DstGc))
	put32(b[12:], ValueMask)
	c.sendRequest(b)
}

func (c *Conn) SetDashes(Gc Id, DashOffset uint16, Dashes []byte) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Dashes) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 58
	put32(b[4:], uint32(Gc))
	put16(b[8:], DashOffset)
	put16(b[10:], uint16(len(Dashes)))
	c.sendRequest(b)
	c.sendBytes(Dashes[0:len(Dashes)])
}

const (
	ClipOrderingUnsorted = 0
	ClipOrderingYSorted  = 1
	ClipOrderingYXSorted = 2
	ClipOrderingYXBanded = 3
)

func (c *Conn) SetClipRectangles(Ordering byte, Gc Id, ClipXOrigin int16, ClipYOrigin int16, Rectangles []Rectangle) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Rectangles) * 8)
	put16(b[2:], uint16(n/4))
	b[0] = 59
	b[1] = Ordering
	put32(b[4:], uint32(Gc))
	put16(b[8:], uint16(ClipXOrigin))
	put16(b[10:], uint16(ClipYOrigin))
	c.sendRequest(b)
	c.sendRectangleList(Rectangles, len(Rectangles))
}

func (c *Conn) FreeGC(Gc Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 60
	put32(b[4:], uint32(Gc))
	c.sendRequest(b)
}

func (c *Conn) ClearArea(Exposures byte, Window Id, X int16, Y int16, Width uint16, Height uint16) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 61
	b[1] = Exposures
	put32(b[4:], uint32(Window))
	put16(b[8:], uint16(X))
	put16(b[10:], uint16(Y))
	put16(b[12:], Width)
	put16(b[14:], Height)
	c.sendRequest(b)
}

func (c *Conn) CopyArea(SrcDrawable Id, DstDrawable Id, Gc Id, SrcX int16, SrcY int16, DstX int16, DstY int16, Width uint16, Height uint16) {
	b := c.scratch[0:28]
	put16(b[2:], 7)
	b[0] = 62
	put32(b[4:], uint32(SrcDrawable))
	put32(b[8:], uint32(DstDrawable))
	put32(b[12:], uint32(Gc))
	put16(b[16:], uint16(SrcX))
	put16(b[18:], uint16(SrcY))
	put16(b[20:], uint16(DstX))
	put16(b[22:], uint16(DstY))
	put16(b[24:], Width)
	put16(b[26:], Height)
	c.sendRequest(b)
}

func (c *Conn) CopyPlane(SrcDrawable Id, DstDrawable Id, Gc Id, SrcX int16, SrcY int16, DstX int16, DstY int16, Width uint16, Height uint16, BitPlane uint32) {
	b := c.scratch[0:32]
	put16(b[2:], 8)
	b[0] = 63
	put32(b[4:], uint32(SrcDrawable))
	put32(b[8:], uint32(DstDrawable))
	put32(b[12:], uint32(Gc))
	put16(b[16:], uint16(SrcX))
	put16(b[18:], uint16(SrcY))
	put16(b[20:], uint16(DstX))
	put16(b[22:], uint16(DstY))
	put16(b[24:], Width)
	put16(b[26:], Height)
	put32(b[28:], BitPlane)
	c.sendRequest(b)
}

const (
	CoordModeOrigin   = 0
	CoordModePrevious = 1
)

func (c *Conn) PolyPoint(CoordinateMode byte, Drawable Id, Gc Id, Points []Point) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Points) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 64
	b[1] = CoordinateMode
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendPointList(Points, len(Points))
}

func (c *Conn) PolyLine(CoordinateMode byte, Drawable Id, Gc Id, Points []Point) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Points) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 65
	b[1] = CoordinateMode
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendPointList(Points, len(Points))
}

type Segment struct {
	X1 int16
	Y1 int16
	X2 int16
	Y2 int16
}

func getSegment(b []byte, v *Segment) int {
	v.X1 = int16(get16(b[0:]))
	v.Y1 = int16(get16(b[2:]))
	v.X2 = int16(get16(b[4:]))
	v.Y2 = int16(get16(b[6:]))
	return 8
}

func (c *Conn) sendSegmentList(list []Segment, count int) {
	b0 := make([]byte, 8*count)
	for k := 0; k < count; k++ {
		b := b0[k*8:]
		put16(b[0:], uint16(list[k].X1))
		put16(b[2:], uint16(list[k].Y1))
		put16(b[4:], uint16(list[k].X2))
		put16(b[6:], uint16(list[k].Y2))
	}
	c.sendBytes(b0)
}

func (c *Conn) PolySegment(Drawable Id, Gc Id, Segments []Segment) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Segments) * 8)
	put16(b[2:], uint16(n/4))
	b[0] = 66
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendSegmentList(Segments, len(Segments))
}

func (c *Conn) PolyRectangle(Drawable Id, Gc Id, Rectangles []Rectangle) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Rectangles) * 8)
	put16(b[2:], uint16(n/4))
	b[0] = 67
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendRectangleList(Rectangles, len(Rectangles))
}

func (c *Conn) PolyArc(Drawable Id, Gc Id, Arcs []Arc) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Arcs) * 12)
	put16(b[2:], uint16(n/4))
	b[0] = 68
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendArcList(Arcs, len(Arcs))
}

const (
	PolyShapeComplex   = 0
	PolyShapeNonconvex = 1
	PolyShapeConvex    = 2
)

func (c *Conn) FillPoly(Drawable Id, Gc Id, Shape byte, CoordinateMode byte, Points []Point) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(Points) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 69
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	b[12] = Shape
	b[13] = CoordinateMode
	c.sendRequest(b)
	c.sendPointList(Points, len(Points))
}

func (c *Conn) PolyFillRectangle(Drawable Id, Gc Id, Rectangles []Rectangle) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Rectangles) * 8)
	put16(b[2:], uint16(n/4))
	b[0] = 70
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendRectangleList(Rectangles, len(Rectangles))
}

func (c *Conn) PolyFillArc(Drawable Id, Gc Id, Arcs []Arc) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Arcs) * 12)
	put16(b[2:], uint16(n/4))
	b[0] = 71
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	c.sendRequest(b)
	c.sendArcList(Arcs, len(Arcs))
}

const (
	ImageFormatXYBitmap = 0
	ImageFormatXYPixmap = 1
	ImageFormatZPixmap  = 2
)

func (c *Conn) PutImage(Format byte, Drawable Id, Gc Id, Width uint16, Height uint16, DstX int16, DstY int16, LeftPad byte, Depth byte, Data []byte) {
	b := c.scratch[0:24]
	n := 24
	n += pad(len(Data) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 72
	b[1] = Format
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	put16(b[12:], Width)
	put16(b[14:], Height)
	put16(b[16:], uint16(DstX))
	put16(b[18:], uint16(DstY))
	b[20] = LeftPad
	b[21] = Depth
	c.sendRequest(b)
	c.sendBytes(Data[0:len(Data)])
}

func (c *Conn) GetImageRequest(Format byte, Drawable Id, X int16, Y int16, Width uint16, Height uint16, PlaneMask uint32) Cookie {
	b := c.scratch[0:20]
	put16(b[2:], 5)
	b[0] = 73
	b[1] = Format
	put32(b[4:], uint32(Drawable))
	put16(b[8:], uint16(X))
	put16(b[10:], uint16(Y))
	put16(b[12:], Width)
	put16(b[14:], Height)
	put32(b[16:], PlaneMask)
	return c.sendRequest(b)
}

func (c *Conn) GetImage(Format byte, Drawable Id, X int16, Y int16, Width uint16, Height uint16, PlaneMask uint32) (*GetImageReply, os.Error) {
	return c.GetImageReply(c.GetImageRequest(Format, Drawable, X, Y, Width, Height, PlaneMask))
}

type GetImageReply struct {
	Depth  byte
	Length uint32
	Visual Id
	Data   []byte
}

func (c *Conn) GetImageReply(cookie Cookie) (*GetImageReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetImageReply)
	v.Depth = b[1]
	v.Length = get32(b[4:])
	v.Visual = Id(get32(b[8:]))
	offset := 32
	v.Data = make([]byte, (int(v.Length) * 4))
	copy(v.Data[0:len(v.Data)], b[offset:])
	offset += len(v.Data) * 1
	return v, nil
}

func (c *Conn) PolyText8(Drawable Id, Gc Id, X int16, Y int16, Items []byte) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(Items) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 74
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	c.sendRequest(b)
	c.sendBytes(Items[0:len(Items)])
}

func (c *Conn) PolyText16(Drawable Id, Gc Id, X int16, Y int16, Items []byte) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(Items) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 75
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	c.sendRequest(b)
	c.sendBytes(Items[0:len(Items)])
}

func (c *Conn) ImageText8(Drawable Id, Gc Id, X int16, Y int16, String []byte) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(String) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 76
	b[1] = byte(len(String))
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	c.sendRequest(b)
	c.sendBytes(String[0:len(String)])
}

func (c *Conn) ImageText16(Drawable Id, Gc Id, X int16, Y int16, String []Char2b) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(String) * 2)
	put16(b[2:], uint16(n/4))
	b[0] = 77
	b[1] = byte(len(String))
	put32(b[4:], uint32(Drawable))
	put32(b[8:], uint32(Gc))
	put16(b[12:], uint16(X))
	put16(b[14:], uint16(Y))
	c.sendRequest(b)
	c.sendChar2bList(String, len(String))
}

const (
	ColormapAllocNone = 0
	ColormapAllocAll  = 1
)

func (c *Conn) CreateColormap(Alloc byte, Mid Id, Window Id, Visual Id) {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 78
	b[1] = Alloc
	put32(b[4:], uint32(Mid))
	put32(b[8:], uint32(Window))
	put32(b[12:], uint32(Visual))
	c.sendRequest(b)
}

func (c *Conn) FreeColormap(Cmap Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 79
	put32(b[4:], uint32(Cmap))
	c.sendRequest(b)
}

func (c *Conn) CopyColormapAndFree(Mid Id, SrcCmap Id) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 80
	put32(b[4:], uint32(Mid))
	put32(b[8:], uint32(SrcCmap))
	c.sendRequest(b)
}

func (c *Conn) InstallColormap(Cmap Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 81
	put32(b[4:], uint32(Cmap))
	c.sendRequest(b)
}

func (c *Conn) UninstallColormap(Cmap Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 82
	put32(b[4:], uint32(Cmap))
	c.sendRequest(b)
}

func (c *Conn) ListInstalledColormapsRequest(Window Id) Cookie {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 83
	put32(b[4:], uint32(Window))
	return c.sendRequest(b)
}

func (c *Conn) ListInstalledColormaps(Window Id) (*ListInstalledColormapsReply, os.Error) {
	return c.ListInstalledColormapsReply(c.ListInstalledColormapsRequest(Window))
}

type ListInstalledColormapsReply struct {
	CmapsLen uint16
	Cmaps    []Id
}

func (c *Conn) ListInstalledColormapsReply(cookie Cookie) (*ListInstalledColormapsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListInstalledColormapsReply)
	v.CmapsLen = get16(b[8:])
	offset := 32
	v.Cmaps = make([]Id, int(v.CmapsLen))
	for i := 0; i < len(v.Cmaps); i++ {
		v.Cmaps[i] = Id(get32(b[offset+i*4:]))
	}
	offset += len(v.Cmaps) * 4
	return v, nil
}

func (c *Conn) AllocColorRequest(Cmap Id, Red uint16, Green uint16, Blue uint16) Cookie {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 84
	put32(b[4:], uint32(Cmap))
	put16(b[8:], Red)
	put16(b[10:], Green)
	put16(b[12:], Blue)
	return c.sendRequest(b)
}

func (c *Conn) AllocColor(Cmap Id, Red uint16, Green uint16, Blue uint16) (*AllocColorReply, os.Error) {
	return c.AllocColorReply(c.AllocColorRequest(Cmap, Red, Green, Blue))
}

type AllocColorReply struct {
	Red   uint16
	Green uint16
	Blue  uint16
	Pixel uint32
}

func (c *Conn) AllocColorReply(cookie Cookie) (*AllocColorReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(AllocColorReply)
	v.Red = get16(b[8:])
	v.Green = get16(b[10:])
	v.Blue = get16(b[12:])
	v.Pixel = get32(b[16:])
	return v, nil
}

func (c *Conn) AllocNamedColorRequest(Cmap Id, Name string) Cookie {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 85
	put32(b[4:], uint32(Cmap))
	put16(b[8:], uint16(len(Name)))
	cookie := c.sendRequest(b)
	c.sendString(Name)
	return cookie
}

func (c *Conn) AllocNamedColor(Cmap Id, Name string) (*AllocNamedColorReply, os.Error) {
	return c.AllocNamedColorReply(c.AllocNamedColorRequest(Cmap, Name))
}

type AllocNamedColorReply struct {
	Pixel       uint32
	ExactRed    uint16
	ExactGreen  uint16
	ExactBlue   uint16
	VisualRed   uint16
	VisualGreen uint16
	VisualBlue  uint16
}

func (c *Conn) AllocNamedColorReply(cookie Cookie) (*AllocNamedColorReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(AllocNamedColorReply)
	v.Pixel = get32(b[8:])
	v.ExactRed = get16(b[12:])
	v.ExactGreen = get16(b[14:])
	v.ExactBlue = get16(b[16:])
	v.VisualRed = get16(b[18:])
	v.VisualGreen = get16(b[20:])
	v.VisualBlue = get16(b[22:])
	return v, nil
}

func (c *Conn) AllocColorCellsRequest(Contiguous byte, Cmap Id, Colors uint16, Planes uint16) Cookie {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 86
	b[1] = Contiguous
	put32(b[4:], uint32(Cmap))
	put16(b[8:], Colors)
	put16(b[10:], Planes)
	return c.sendRequest(b)
}

func (c *Conn) AllocColorCells(Contiguous byte, Cmap Id, Colors uint16, Planes uint16) (*AllocColorCellsReply, os.Error) {
	return c.AllocColorCellsReply(c.AllocColorCellsRequest(Contiguous, Cmap, Colors, Planes))
}

type AllocColorCellsReply struct {
	PixelsLen uint16
	MasksLen  uint16
	Pixels    []uint32
	Masks     []uint32
}

func (c *Conn) AllocColorCellsReply(cookie Cookie) (*AllocColorCellsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(AllocColorCellsReply)
	v.PixelsLen = get16(b[8:])
	v.MasksLen = get16(b[10:])
	offset := 32
	v.Pixels = make([]uint32, int(v.PixelsLen))
	for i := 0; i < len(v.Pixels); i++ {
		v.Pixels[i] = get32(b[offset+i*4:])
	}
	offset += len(v.Pixels) * 4
	offset = pad(offset)
	v.Masks = make([]uint32, int(v.MasksLen))
	for i := 0; i < len(v.Masks); i++ {
		v.Masks[i] = get32(b[offset+i*4:])
	}
	offset += len(v.Masks) * 4
	return v, nil
}

func (c *Conn) AllocColorPlanesRequest(Contiguous byte, Cmap Id, Colors uint16, Reds uint16, Greens uint16, Blues uint16) Cookie {
	b := c.scratch[0:16]
	put16(b[2:], 4)
	b[0] = 87
	b[1] = Contiguous
	put32(b[4:], uint32(Cmap))
	put16(b[8:], Colors)
	put16(b[10:], Reds)
	put16(b[12:], Greens)
	put16(b[14:], Blues)
	return c.sendRequest(b)
}

func (c *Conn) AllocColorPlanes(Contiguous byte, Cmap Id, Colors uint16, Reds uint16, Greens uint16, Blues uint16) (*AllocColorPlanesReply, os.Error) {
	return c.AllocColorPlanesReply(c.AllocColorPlanesRequest(Contiguous, Cmap, Colors, Reds, Greens, Blues))
}

type AllocColorPlanesReply struct {
	PixelsLen uint16
	RedMask   uint32
	GreenMask uint32
	BlueMask  uint32
	Pixels    []uint32
}

func (c *Conn) AllocColorPlanesReply(cookie Cookie) (*AllocColorPlanesReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(AllocColorPlanesReply)
	v.PixelsLen = get16(b[8:])
	v.RedMask = get32(b[12:])
	v.GreenMask = get32(b[16:])
	v.BlueMask = get32(b[20:])
	offset := 32
	v.Pixels = make([]uint32, int(v.PixelsLen))
	for i := 0; i < len(v.Pixels); i++ {
		v.Pixels[i] = get32(b[offset+i*4:])
	}
	offset += len(v.Pixels) * 4
	return v, nil
}

func (c *Conn) FreeColors(Cmap Id, PlaneMask uint32, Pixels []uint32) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Pixels) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 88
	put32(b[4:], uint32(Cmap))
	put32(b[8:], PlaneMask)
	c.sendRequest(b)
	c.sendUInt32List(Pixels[0:len(Pixels)])
}

const (
	ColorFlagRed   = 1
	ColorFlagGreen = 2
	ColorFlagBlue  = 4
)

type Coloritem struct {
	Pixel uint32
	Red   uint16
	Green uint16
	Blue  uint16
	Flags byte
}

func getColoritem(b []byte, v *Coloritem) int {
	v.Pixel = get32(b[0:])
	v.Red = get16(b[4:])
	v.Green = get16(b[6:])
	v.Blue = get16(b[8:])
	v.Flags = b[10]
	return 12
}

func (c *Conn) sendColoritemList(list []Coloritem, count int) {
	b0 := make([]byte, 12*count)
	for k := 0; k < count; k++ {
		b := b0[k*12:]
		put32(b[0:], list[k].Pixel)
		put16(b[4:], list[k].Red)
		put16(b[6:], list[k].Green)
		put16(b[8:], list[k].Blue)
		b[10] = list[k].Flags
	}
	c.sendBytes(b0)
}

func (c *Conn) StoreColors(Cmap Id, Items []Coloritem) {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Items) * 12)
	put16(b[2:], uint16(n/4))
	b[0] = 89
	put32(b[4:], uint32(Cmap))
	c.sendRequest(b)
	c.sendColoritemList(Items, len(Items))
}

func (c *Conn) StoreNamedColor(Flags byte, Cmap Id, Pixel uint32, Name string) {
	b := c.scratch[0:16]
	n := 16
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 90
	b[1] = Flags
	put32(b[4:], uint32(Cmap))
	put32(b[8:], Pixel)
	put16(b[12:], uint16(len(Name)))
	c.sendRequest(b)
	c.sendString(Name)
}

type Rgb struct {
	Red   uint16
	Green uint16
	Blue  uint16
}

func getRgb(b []byte, v *Rgb) int {
	v.Red = get16(b[0:])
	v.Green = get16(b[2:])
	v.Blue = get16(b[4:])
	return 8
}

func (c *Conn) sendRgbList(list []Rgb, count int) {
	b0 := make([]byte, 8*count)
	for k := 0; k < count; k++ {
		b := b0[k*8:]
		put16(b[0:], list[k].Red)
		put16(b[2:], list[k].Green)
		put16(b[4:], list[k].Blue)
	}
	c.sendBytes(b0)
}

func (c *Conn) QueryColorsRequest(Cmap Id, Pixels []uint32) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Pixels) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 91
	put32(b[4:], uint32(Cmap))
	cookie := c.sendRequest(b)
	c.sendUInt32List(Pixels[0:len(Pixels)])
	return cookie
}

func (c *Conn) QueryColors(Cmap Id, Pixels []uint32) (*QueryColorsReply, os.Error) {
	return c.QueryColorsReply(c.QueryColorsRequest(Cmap, Pixels))
}

type QueryColorsReply struct {
	ColorsLen uint16
	Colors    []Rgb
}

func (c *Conn) QueryColorsReply(cookie Cookie) (*QueryColorsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryColorsReply)
	v.ColorsLen = get16(b[8:])
	offset := 32
	v.Colors = make([]Rgb, int(v.ColorsLen))
	for i := 0; i < int(v.ColorsLen); i++ {
		offset += getRgb(b[offset:], &v.Colors[i])
	}
	return v, nil
}

func (c *Conn) LookupColorRequest(Cmap Id, Name string) Cookie {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 92
	put32(b[4:], uint32(Cmap))
	put16(b[8:], uint16(len(Name)))
	cookie := c.sendRequest(b)
	c.sendString(Name)
	return cookie
}

func (c *Conn) LookupColor(Cmap Id, Name string) (*LookupColorReply, os.Error) {
	return c.LookupColorReply(c.LookupColorRequest(Cmap, Name))
}

type LookupColorReply struct {
	ExactRed    uint16
	ExactGreen  uint16
	ExactBlue   uint16
	VisualRed   uint16
	VisualGreen uint16
	VisualBlue  uint16
}

func (c *Conn) LookupColorReply(cookie Cookie) (*LookupColorReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(LookupColorReply)
	v.ExactRed = get16(b[8:])
	v.ExactGreen = get16(b[10:])
	v.ExactBlue = get16(b[12:])
	v.VisualRed = get16(b[14:])
	v.VisualGreen = get16(b[16:])
	v.VisualBlue = get16(b[18:])
	return v, nil
}

const (
	PixmapNone = 0
)

func (c *Conn) CreateCursor(Cid Id, Source Id, Mask Id, ForeRed uint16, ForeGreen uint16, ForeBlue uint16, BackRed uint16, BackGreen uint16, BackBlue uint16, X uint16, Y uint16) {
	b := c.scratch[0:32]
	put16(b[2:], 8)
	b[0] = 93
	put32(b[4:], uint32(Cid))
	put32(b[8:], uint32(Source))
	put32(b[12:], uint32(Mask))
	put16(b[16:], ForeRed)
	put16(b[18:], ForeGreen)
	put16(b[20:], ForeBlue)
	put16(b[22:], BackRed)
	put16(b[24:], BackGreen)
	put16(b[26:], BackBlue)
	put16(b[28:], X)
	put16(b[30:], Y)
	c.sendRequest(b)
}

const (
	FontNone = 0
)

func (c *Conn) CreateGlyphCursor(Cid Id, SourceFont Id, MaskFont Id, SourceChar uint16, MaskChar uint16, ForeRed uint16, ForeGreen uint16, ForeBlue uint16, BackRed uint16, BackGreen uint16, BackBlue uint16) {
	b := c.scratch[0:32]
	put16(b[2:], 8)
	b[0] = 94
	put32(b[4:], uint32(Cid))
	put32(b[8:], uint32(SourceFont))
	put32(b[12:], uint32(MaskFont))
	put16(b[16:], SourceChar)
	put16(b[18:], MaskChar)
	put16(b[20:], ForeRed)
	put16(b[22:], ForeGreen)
	put16(b[24:], ForeBlue)
	put16(b[26:], BackRed)
	put16(b[28:], BackGreen)
	put16(b[30:], BackBlue)
	c.sendRequest(b)
}

func (c *Conn) FreeCursor(Cursor Id) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 95
	put32(b[4:], uint32(Cursor))
	c.sendRequest(b)
}

func (c *Conn) RecolorCursor(Cursor Id, ForeRed uint16, ForeGreen uint16, ForeBlue uint16, BackRed uint16, BackGreen uint16, BackBlue uint16) {
	b := c.scratch[0:20]
	put16(b[2:], 5)
	b[0] = 96
	put32(b[4:], uint32(Cursor))
	put16(b[8:], ForeRed)
	put16(b[10:], ForeGreen)
	put16(b[12:], ForeBlue)
	put16(b[14:], BackRed)
	put16(b[16:], BackGreen)
	put16(b[18:], BackBlue)
	c.sendRequest(b)
}

const (
	QueryShapeOfLargestCursor  = 0
	QueryShapeOfFastestTile    = 1
	QueryShapeOfFastestStipple = 2
)

func (c *Conn) QueryBestSizeRequest(Class byte, Drawable Id, Width uint16, Height uint16) Cookie {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 97
	b[1] = Class
	put32(b[4:], uint32(Drawable))
	put16(b[8:], Width)
	put16(b[10:], Height)
	return c.sendRequest(b)
}

func (c *Conn) QueryBestSize(Class byte, Drawable Id, Width uint16, Height uint16) (*QueryBestSizeReply, os.Error) {
	return c.QueryBestSizeReply(c.QueryBestSizeRequest(Class, Drawable, Width, Height))
}

type QueryBestSizeReply struct {
	Width  uint16
	Height uint16
}

func (c *Conn) QueryBestSizeReply(cookie Cookie) (*QueryBestSizeReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryBestSizeReply)
	v.Width = get16(b[8:])
	v.Height = get16(b[10:])
	return v, nil
}

func (c *Conn) QueryExtensionRequest(Name string) Cookie {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Name) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 98
	put16(b[4:], uint16(len(Name)))
	cookie := c.sendRequest(b)
	c.sendString(Name)
	return cookie
}

func (c *Conn) QueryExtension(Name string) (*QueryExtensionReply, os.Error) {
	return c.QueryExtensionReply(c.QueryExtensionRequest(Name))
}

type QueryExtensionReply struct {
	Present     byte
	MajorOpcode byte
	FirstEvent  byte
	FirstError  byte
}

func (c *Conn) QueryExtensionReply(cookie Cookie) (*QueryExtensionReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(QueryExtensionReply)
	v.Present = b[8]
	v.MajorOpcode = b[9]
	v.FirstEvent = b[10]
	v.FirstError = b[11]
	return v, nil
}

func (c *Conn) ListExtensionsRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 99
	return c.sendRequest(b)
}

func (c *Conn) ListExtensions() (*ListExtensionsReply, os.Error) {
	return c.ListExtensionsReply(c.ListExtensionsRequest())
}

type ListExtensionsReply struct {
	NamesLen byte
	Names    []Str
}

func (c *Conn) ListExtensionsReply(cookie Cookie) (*ListExtensionsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListExtensionsReply)
	v.NamesLen = b[1]
	offset := 32
	v.Names = make([]Str, int(v.NamesLen))
	for i := 0; i < int(v.NamesLen); i++ {
		offset += getStr(b[offset:], &v.Names[i])
	}
	return v, nil
}

func (c *Conn) ChangeKeyboardMapping(KeycodeCount byte, FirstKeycode byte, KeysymsPerKeycode byte, Keysyms []Keysym) {
	b := c.scratch[0:6]
	n := 6
	n += pad((int(KeycodeCount) * int(KeysymsPerKeycode)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 100
	b[1] = KeycodeCount
	b[4] = FirstKeycode
	b[5] = KeysymsPerKeycode
	c.sendRequest(b)
	c.sendKeysymList(Keysyms, (int(KeycodeCount) * int(KeysymsPerKeycode)))
}

func (c *Conn) GetKeyboardMappingRequest(FirstKeycode byte, Count byte) Cookie {
	b := c.scratch[0:6]
	put16(b[2:], 1)
	b[0] = 101
	b[4] = FirstKeycode
	b[5] = Count
	return c.sendRequest(b)
}

func (c *Conn) GetKeyboardMapping(FirstKeycode byte, Count byte) (*GetKeyboardMappingReply, os.Error) {
	return c.GetKeyboardMappingReply(c.GetKeyboardMappingRequest(FirstKeycode, Count))
}

type GetKeyboardMappingReply struct {
	KeysymsPerKeycode byte
	Length            uint32
	Keysyms           []Keysym
}

func (c *Conn) GetKeyboardMappingReply(cookie Cookie) (*GetKeyboardMappingReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetKeyboardMappingReply)
	v.KeysymsPerKeycode = b[1]
	v.Length = get32(b[4:])
	offset := 32
	v.Keysyms = make([]Keysym, int(v.Length))
	for i := 0; i < len(v.Keysyms); i++ {
		v.Keysyms[i] = Keysym(get32(b[offset+i*4:]))
	}
	offset += len(v.Keysyms) * 4
	return v, nil
}

const (
	KBKeyClickPercent = 1
	KBBellPercent     = 2
	KBBellPitch       = 4
	KBBellDuration    = 8
	KBLed             = 16
	KBLedMode         = 32
	KBKey             = 64
	KBAutoRepeatMode  = 128
)

const (
	LedModeOff = 0
	LedModeOn  = 1
)

const (
	AutoRepeatModeOff     = 0
	AutoRepeatModeOn      = 1
	AutoRepeatModeDefault = 2
)

func (c *Conn) ChangeKeyboardControl(ValueMask uint32, ValueList []uint32) {
	b := c.scratch[0:8]
	n := 8
	n += pad(popCount(int(ValueMask)) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 102
	put32(b[4:], ValueMask)
	c.sendRequest(b)
	c.sendUInt32List(ValueList[0:popCount(int(ValueMask))])
}

func (c *Conn) GetKeyboardControlRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 103
	return c.sendRequest(b)
}

func (c *Conn) GetKeyboardControl() (*GetKeyboardControlReply, os.Error) {
	return c.GetKeyboardControlReply(c.GetKeyboardControlRequest())
}

type GetKeyboardControlReply struct {
	GlobalAutoRepeat byte
	LedMask          uint32
	KeyClickPercent  byte
	BellPercent      byte
	BellPitch        uint16
	BellDuration     uint16
	AutoRepeats      [32]byte
}

func (c *Conn) GetKeyboardControlReply(cookie Cookie) (*GetKeyboardControlReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetKeyboardControlReply)
	v.GlobalAutoRepeat = b[1]
	v.LedMask = get32(b[8:])
	v.KeyClickPercent = b[12]
	v.BellPercent = b[13]
	v.BellPitch = get16(b[14:])
	v.BellDuration = get16(b[16:])
	copy(v.AutoRepeats[0:32], b[20:])
	return v, nil
}

func (c *Conn) Bell(Percent int8) {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 104
	b[1] = byte(Percent)
	c.sendRequest(b)
}

func (c *Conn) ChangePointerControl(AccelerationNumerator int16, AccelerationDenominator int16, Threshold int16, DoAcceleration byte, DoThreshold byte) {
	b := c.scratch[0:12]
	put16(b[2:], 3)
	b[0] = 105
	put16(b[4:], uint16(AccelerationNumerator))
	put16(b[6:], uint16(AccelerationDenominator))
	put16(b[8:], uint16(Threshold))
	b[10] = DoAcceleration
	b[11] = DoThreshold
	c.sendRequest(b)
}

func (c *Conn) GetPointerControlRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 106
	return c.sendRequest(b)
}

func (c *Conn) GetPointerControl() (*GetPointerControlReply, os.Error) {
	return c.GetPointerControlReply(c.GetPointerControlRequest())
}

type GetPointerControlReply struct {
	AccelerationNumerator   uint16
	AccelerationDenominator uint16
	Threshold               uint16
}

func (c *Conn) GetPointerControlReply(cookie Cookie) (*GetPointerControlReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetPointerControlReply)
	v.AccelerationNumerator = get16(b[8:])
	v.AccelerationDenominator = get16(b[10:])
	v.Threshold = get16(b[12:])
	return v, nil
}

const (
	BlankingNotPreferred = 0
	BlankingPreferred    = 1
	BlankingDefault      = 2
)

const (
	ExposuresNotAllowed = 0
	ExposuresAllowed    = 1
	ExposuresDefault    = 2
)

func (c *Conn) SetScreenSaver(Timeout int16, Interval int16, PreferBlanking byte, AllowExposures byte) {
	b := c.scratch[0:10]
	put16(b[2:], 2)
	b[0] = 107
	put16(b[4:], uint16(Timeout))
	put16(b[6:], uint16(Interval))
	b[8] = PreferBlanking
	b[9] = AllowExposures
	c.sendRequest(b)
}

func (c *Conn) GetScreenSaverRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 108
	return c.sendRequest(b)
}

func (c *Conn) GetScreenSaver() (*GetScreenSaverReply, os.Error) {
	return c.GetScreenSaverReply(c.GetScreenSaverRequest())
}

type GetScreenSaverReply struct {
	Timeout        uint16
	Interval       uint16
	PreferBlanking byte
	AllowExposures byte
}

func (c *Conn) GetScreenSaverReply(cookie Cookie) (*GetScreenSaverReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetScreenSaverReply)
	v.Timeout = get16(b[8:])
	v.Interval = get16(b[10:])
	v.PreferBlanking = b[12]
	v.AllowExposures = b[13]
	return v, nil
}

const (
	HostModeInsert = 0
	HostModeDelete = 1
)

const (
	FamilyInternet          = 0
	FamilyDECnet            = 1
	FamilyChaos             = 2
	FamilyServerInterpreted = 5
	FamilyInternet6         = 6
)

func (c *Conn) ChangeHosts(Mode byte, Family byte, Address []byte) {
	b := c.scratch[0:8]
	n := 8
	n += pad(len(Address) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 109
	b[1] = Mode
	b[4] = Family
	put16(b[6:], uint16(len(Address)))
	c.sendRequest(b)
	c.sendBytes(Address[0:len(Address)])
}

type Host struct {
	Family     byte
	AddressLen uint16
	Address    []byte
}

func getHost(b []byte, v *Host) int {
	v.Family = b[0]
	v.AddressLen = get16(b[2:])
	offset := 4
	v.Address = make([]byte, int(v.AddressLen))
	copy(v.Address[0:len(v.Address)], b[offset:])
	offset += len(v.Address) * 1
	return offset
}

// omitting variable length sendHost

func (c *Conn) ListHostsRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 110
	return c.sendRequest(b)
}

func (c *Conn) ListHosts() (*ListHostsReply, os.Error) {
	return c.ListHostsReply(c.ListHostsRequest())
}

type ListHostsReply struct {
	Mode     byte
	HostsLen uint16
	Hosts    []Host
}

func (c *Conn) ListHostsReply(cookie Cookie) (*ListHostsReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(ListHostsReply)
	v.Mode = b[1]
	v.HostsLen = get16(b[8:])
	offset := 32
	v.Hosts = make([]Host, int(v.HostsLen))
	for i := 0; i < int(v.HostsLen); i++ {
		offset += getHost(b[offset:], &v.Hosts[i])
	}
	return v, nil
}

const (
	AccessControlDisable = 0
	AccessControlEnable  = 1
)

func (c *Conn) SetAccessControl(Mode byte) {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 111
	b[1] = Mode
	c.sendRequest(b)
}

const (
	CloseDownDestroyAll      = 0
	CloseDownRetainPermanent = 1
	CloseDownRetainTemporary = 2
)

func (c *Conn) SetCloseDownMode(Mode byte) {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 112
	b[1] = Mode
	c.sendRequest(b)
}

const (
	KillAllTemporary = 0
)

func (c *Conn) KillClient(Resource uint32) {
	b := c.scratch[0:8]
	put16(b[2:], 2)
	b[0] = 113
	put32(b[4:], Resource)
	c.sendRequest(b)
}

func (c *Conn) RotateProperties(Window Id, Delta int16, Atoms []Id) {
	b := c.scratch[0:12]
	n := 12
	n += pad(len(Atoms) * 4)
	put16(b[2:], uint16(n/4))
	b[0] = 114
	put32(b[4:], uint32(Window))
	put16(b[8:], uint16(len(Atoms)))
	put16(b[10:], uint16(Delta))
	c.sendRequest(b)
	c.sendIdList(Atoms, len(Atoms))
}

const (
	ScreenSaverReset  = 0
	ScreenSaverActive = 1
)

func (c *Conn) ForceScreenSaver(Mode byte) {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 115
	b[1] = Mode
	c.sendRequest(b)
}

const (
	MappingStatusSuccess = 0
	MappingStatusBusy    = 1
	MappingStatusFailure = 2
)

func (c *Conn) SetPointerMappingRequest(Map []byte) Cookie {
	b := c.scratch[0:4]
	n := 4
	n += pad(len(Map) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 116
	b[1] = byte(len(Map))
	cookie := c.sendRequest(b)
	c.sendBytes(Map[0:len(Map)])
	return cookie
}

func (c *Conn) SetPointerMapping(Map []byte) (*SetPointerMappingReply, os.Error) {
	return c.SetPointerMappingReply(c.SetPointerMappingRequest(Map))
}

type SetPointerMappingReply struct {
	Status byte
}

func (c *Conn) SetPointerMappingReply(cookie Cookie) (*SetPointerMappingReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(SetPointerMappingReply)
	v.Status = b[1]
	return v, nil
}

func (c *Conn) GetPointerMappingRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 117
	return c.sendRequest(b)
}

func (c *Conn) GetPointerMapping() (*GetPointerMappingReply, os.Error) {
	return c.GetPointerMappingReply(c.GetPointerMappingRequest())
}

type GetPointerMappingReply struct {
	MapLen byte
	Map    []byte
}

func (c *Conn) GetPointerMappingReply(cookie Cookie) (*GetPointerMappingReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetPointerMappingReply)
	v.MapLen = b[1]
	offset := 32
	v.Map = make([]byte, int(v.MapLen))
	copy(v.Map[0:len(v.Map)], b[offset:])
	offset += len(v.Map) * 1
	return v, nil
}

const (
	MapIndexShift   = 0
	MapIndexLock    = 1
	MapIndexControl = 2
	MapIndex1       = 3
	MapIndex2       = 4
	MapIndex3       = 5
	MapIndex4       = 6
	MapIndex5       = 7
)

func (c *Conn) SetModifierMappingRequest(KeycodesPerModifier byte, Keycodes []byte) Cookie {
	b := c.scratch[0:4]
	n := 4
	n += pad((int(KeycodesPerModifier) * 8) * 1)
	put16(b[2:], uint16(n/4))
	b[0] = 118
	b[1] = KeycodesPerModifier
	cookie := c.sendRequest(b)
	c.sendBytes(Keycodes[0:(int(KeycodesPerModifier) * 8)])
	return cookie
}

func (c *Conn) SetModifierMapping(KeycodesPerModifier byte, Keycodes []byte) (*SetModifierMappingReply, os.Error) {
	return c.SetModifierMappingReply(c.SetModifierMappingRequest(KeycodesPerModifier, Keycodes))
}

type SetModifierMappingReply struct {
	Status byte
}

func (c *Conn) SetModifierMappingReply(cookie Cookie) (*SetModifierMappingReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(SetModifierMappingReply)
	v.Status = b[1]
	return v, nil
}

func (c *Conn) GetModifierMappingRequest() Cookie {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 119
	return c.sendRequest(b)
}

func (c *Conn) GetModifierMapping() (*GetModifierMappingReply, os.Error) {
	return c.GetModifierMappingReply(c.GetModifierMappingRequest())
}

type GetModifierMappingReply struct {
	KeycodesPerModifier byte
	Keycodes            []byte
}

func (c *Conn) GetModifierMappingReply(cookie Cookie) (*GetModifierMappingReply, os.Error) {
	b, error := c.waitForReply(cookie)
	if error != nil {
		return nil, error
	}
	v := new(GetModifierMappingReply)
	v.KeycodesPerModifier = b[1]
	offset := 32
	v.Keycodes = make([]byte, (int(v.KeycodesPerModifier) * 8))
	copy(v.Keycodes[0:len(v.Keycodes)], b[offset:])
	offset += len(v.Keycodes) * 1
	return v, nil
}

func (c *Conn) NoOperation() {
	b := c.scratch[0:4]
	put16(b[2:], 1)
	b[0] = 127
	c.sendRequest(b)
}

func parseEvent(buf []byte) (Event, os.Error) {
	switch buf[0] {
	case KeyPress:
		return getKeyPressEvent(buf), nil
	case KeyRelease:
		return getKeyReleaseEvent(buf), nil
	case ButtonPress:
		return getButtonPressEvent(buf), nil
	case ButtonRelease:
		return getButtonReleaseEvent(buf), nil
	case MotionNotify:
		return getMotionNotifyEvent(buf), nil
	case EnterNotify:
		return getEnterNotifyEvent(buf), nil
	case LeaveNotify:
		return getLeaveNotifyEvent(buf), nil
	case FocusIn:
		return getFocusInEvent(buf), nil
	case FocusOut:
		return getFocusOutEvent(buf), nil
	case KeymapNotify:
		return getKeymapNotifyEvent(buf), nil
	case Expose:
		return getExposeEvent(buf), nil
	case GraphicsExposure:
		return getGraphicsExposureEvent(buf), nil
	case NoExposure:
		return getNoExposureEvent(buf), nil
	case VisibilityNotify:
		return getVisibilityNotifyEvent(buf), nil
	case CreateNotify:
		return getCreateNotifyEvent(buf), nil
	case DestroyNotify:
		return getDestroyNotifyEvent(buf), nil
	case UnmapNotify:
		return getUnmapNotifyEvent(buf), nil
	case MapNotify:
		return getMapNotifyEvent(buf), nil
	case MapRequest:
		return getMapRequestEvent(buf), nil
	case ReparentNotify:
		return getReparentNotifyEvent(buf), nil
	case ConfigureNotify:
		return getConfigureNotifyEvent(buf), nil
	case ConfigureRequest:
		return getConfigureRequestEvent(buf), nil
	case GravityNotify:
		return getGravityNotifyEvent(buf), nil
	case ResizeRequest:
		return getResizeRequestEvent(buf), nil
	case CirculateNotify:
		return getCirculateNotifyEvent(buf), nil
	case CirculateRequest:
		return getCirculateRequestEvent(buf), nil
	case PropertyNotify:
		return getPropertyNotifyEvent(buf), nil
	case SelectionClear:
		return getSelectionClearEvent(buf), nil
	case SelectionRequest:
		return getSelectionRequestEvent(buf), nil
	case SelectionNotify:
		return getSelectionNotifyEvent(buf), nil
	case ColormapNotify:
		return getColormapNotifyEvent(buf), nil
	case ClientMessage:
		return getClientMessageEvent(buf), nil
	case MappingNotify:
		return getMappingNotifyEvent(buf), nil
	}
	return nil, os.NewError("unknown event type")
}

var errorNames = map[byte]string{
	BadRequest: "Request",
	BadValue: "Value",
	BadWindow: "Window",
	BadPixmap: "Pixmap",
	BadAtom: "Atom",
	BadCursor: "Cursor",
	BadFont: "Font",
	BadMatch: "Match",
	BadDrawable: "Drawable",
	BadAccess: "Access",
	BadAlloc: "Alloc",
	BadColormap: "Colormap",
	BadGContext: "GContext",
	BadIDChoice: "IDChoice",
	BadName: "Name",
	BadLength: "Length",
	BadImplementation: "Implementation",
}
